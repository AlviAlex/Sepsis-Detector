from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import joblib
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and feature means
model = joblib.load("sepsis_xgb_balanced.pkl")
with open('feature_means.json', 'r') as f:
    feature_means = json.load(f)

# Get the list of all features the model was trained on
ALL_MODEL_FEATURES = list(feature_means.keys())

@app.route('/')
def home():
    return "Sepsis Detector API is online and running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the frontend
        user_input = request.get_json()

        # Create a dictionary for a single patient record, starting with the average values
        patient_data = feature_means.copy()

        # Update the dictionary with the values provided by the user
        for key, value in user_input.items():
            if key in patient_data:
                patient_data[key] = value

        # Convert the complete patient data into a DataFrame for the model
        df = pd.DataFrame([patient_data], columns=ALL_MODEL_FEATURES)

        # Predict the probability of sepsis
        probability = model.predict_proba(df)[0, 1]

        return jsonify({'probability': float(probability)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
