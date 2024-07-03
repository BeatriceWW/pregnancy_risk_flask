from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Load your model
model = pickle.load(open('trained_model.sav', 'rb'))

@app.route('/api/predict', methods=['POST'])


@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Pregnancy Risk Prediction API",
        "usage": "Send a POST request to /api/predict with the required data"})
def predict_api():
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
    
    return jsonify({"prediction": prediction})

# Your existing risk_level_prediction function
def risk_level_prediction(input_data):
    # Your existing prediction logic here
    # ...
    return prediction  # Make sure this returns a string or JSON-serializable data


if __name__ == '__main__':
    app.run(debug=True)