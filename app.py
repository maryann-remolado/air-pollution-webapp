from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import json
import numpy as np

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load model and metadata
try:
    model = joblib.load('pollution_risk_model.pkl')
    with open('model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Location mapping
location_mapping = {
    'manila': 0, 'quezon': 1, 'makati': 2, 'taguig': 3,
    'pasig': 4, 'mandaluyong': 5, 'pasay': 6, 'paranaque': 7,
    'laspinas': 8, 'muntinlupa': 9
}

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_accuracy": model_metadata.get('accuracy', 'unknown') if model else 'no model'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        
        # Prepare features
        features = [
            float(data.get('pm25', 0)),
            float(data.get('pm10', 0)),
            float(data.get('no2', 0)),
            float(data.get('so2', 0)),
            float(data.get('co', 0)),
            float(data.get('o3', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            location_mapping.get(data.get('location', 'manila'), 0)
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([features])[0]
        else:
            probabilities = [0.33, 0.34, 0.33]
        
        risk_levels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        risk = risk_levels.get(prediction, 'Moderate')
        confidence = round(probabilities[prediction] * 100, 2)
        
        return jsonify({
            "risk_level": risk,
            "confidence": confidence,
            "probabilities": {
                "Low": round(probabilities[0] * 100, 2),
                "Moderate": round(probabilities[1] * 100, 2),
                "High": round(probabilities[2] * 100, 2)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
