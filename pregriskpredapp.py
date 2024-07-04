from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Load your model
model = pickle.load(open('trained_model.sav', 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        features = [
            float(data['Age']),
            float(data['Body_Temperature']),
            float(data['Heart_Rate']),
            float(data['Systolic_Blood_Pressure']),
            float(data['Diastolic_Blood_Pressure']),
            float(data['BMI']),
            float(data['Blood_Glucose_HbA1c']),
            float(data['Blood_Glucose_Fasting'])
        ]
        
        prediction = risk_level_prediction(features)
        
        return jsonify({
            "prediction": prediction,
            "disclaimer": "This prediction is based on a machine learning model and is not 100% accurate. Always consult with a healthcare professional for medical advice."
         })

    
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def risk_level_prediction(input_data):
    # Convert the input data into a numpy array
    input_data_as_numpy = np.asarray(input_data)

    # Reshape the array since we are predicting for one instance
    input_data_reshape = input_data_as_numpy.reshape(1,-1)

    # Making prediction
    prediction = model.predict(input_data_reshape)


    if prediction[0] == 0:
        return 'High Risk:It is crucial to seek urgent medical attention to ensure the health and safety of both you and your baby. Please consult your healthcare provider immediately.'
    elif prediction[0] == 1:
        return 'Low Risk: Your pregnancy is currently considered low risk. Continue with your regular prenatal check-ups and maintain a healthy lifestyle.'
    elif prediction[0] == 2:
        return 'Medium Risk:It is important to seek medical attention to monitor and manage any potential complications. Please schedule an appointment with your healthcare provider soon.'
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