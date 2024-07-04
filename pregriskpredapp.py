from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import logging

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your model
model = pickle.load(open('trained_model.sav', 'rb'))

def validate_input(features):
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)):
            raise ValueError(f"Input at position {i} must be a number, not {type(feature)}")
        if feature is None or feature == "":
            raise ValueError(f"Input at position {i} cannot be empty")

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        app.logger.info(f"Received data: {data}")

        # Create a mapping of expected keys to possible variations
        key_mapping = {
            "Age": ["age", "Age"],
            "Body_Temperature": ["body_temperature", "bodytemperature", "BodyTemperature", "Body_Temperature"],
            "Heart_Rate": ["heart_rate", "heartrate", "HeartRate", "Heart_Rate"],
            "Systolic_Blood_Pressure": ["systolic_blood_pressure", "systolicbloodpressure", "SystolicBloodPressure", "Systolic_Blood_Pressure"],
            "Diastolic_Blood_Pressure": ["diastolic_blood_pressure", "diastolicbloodpressure", "DiastolicBloodPressure", "Diastolic_Blood_Pressure"],
            "BMI": ["bmi", "BMI"],
            "Blood_Glucose_HbA1c": ["blood_glucose_hba1c", "bloodglucosehba1c", "BloodGlucoseHbA1c", "Blood_Glucose_HbA1c"],
            "Blood_Glucose_Fasting": ["blood_glucose_fasting", "bloodglucosefasting", "BloodGlucoseFasting", "Blood_Glucose_Fasting"]
        }

        # Function to find the correct key
        def find_key(d, key_to_find):
            for expected_key, variations in key_mapping.items():
                if key_to_find in variations:
                    return expected_key
            raise KeyError(f"No matching key found for {key_to_find}")

        # Extract features using the mapping
        features = []
        for key in key_mapping.keys():
            value = data.get(find_key(data, key))
            if value is None or value == "":
                raise ValueError(f"Missing or empty value for {key}")
            features.append(float(value))

        # Validate input
        validate_input(features)
        
        prediction = risk_level_prediction(features)
        
        return jsonify({
            "prediction": prediction,
            "input_data": {key: data[find_key(data, key)] for key in key_mapping.keys()},
            "disclaimer": "This prediction is based on a machine learning model and is not 100% accurate. Always consult with a healthcare professional for medical advice."
         })

    except KeyError as e:
        app.logger.error(f"Missing or incorrect field name: {str(e)}")
        return jsonify({"error": f"Missing or incorrect field name: {str(e)}"}), 400
    except ValueError as e:
        app.logger.error(f"Invalid value: {str(e)}")
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def risk_level_prediction(input_data):
    # Convert the input data into a numpy array
    input_data_as_numpy = np.asarray(input_data)

    # Reshape the array since we are predicting for one instance
    input_data_reshape = input_data_as_numpy.reshape(1,-1)

    # Making prediction
    prediction = model.predict(input_data_reshape)

    if prediction[0] == 0:
        return 'High Risk: It is crucial to seek urgent medical attention to ensure the health and safety of both you and your baby. Please consult your healthcare provider immediately.'
    elif prediction[0] == 1:
        return 'Low Risk: Your pregnancy is currently considered low risk. Continue with your regular prenatal check-ups and maintain a healthy lifestyle.'
    elif prediction[0] == 2:
        return 'Medium Risk: It is important to seek medical attention to monitor and manage any potential complications. Please schedule an appointment with your healthcare provider soon.'
    else:
        return 'Error: Please enter valid data'

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Pregnancy Risk Prediction API",
        "usage": "Send a POST request to /api/predict with the required data"
    })

if __name__ == '__main__':
    app.run(debug=True)