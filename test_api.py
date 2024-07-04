import requests
import json

# Your API endpoint URL
url = "https://pregnancy-risk-flask.onrender.com/api/predict"

# Sample data
data = {
    "Age": 30,
    "Body_Temperature": 98.6,
    "Heart_Rate": 80,
    "Systolic_Blood_Pressure": 120,
    "Diastolic_Blood_Pressure": 80,
    "BMI": 22,
    "Blood_Glucose_HbA1c": 5.7,
    "Blood_Glucose_Fasting": 90
}

# Send POST request
try:
    response = requests.post(url, json=data, timeout=120)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Success! Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response content:")
        print(response.text)
except requests.exceptions.Timeout:
    print("The request timed out. Your API might be taking too long to respond.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")


   