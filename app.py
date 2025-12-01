
# ============================================
# FLASK WEB APP FOR POLLUTION RISK ASSESSMENT
# ============================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'pollution_model.joblib'
ENCODER_PATH = 'target_encoder.joblib'
FEATURES_PATH = 'feature_names.json'

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

with open(FEATURES_PATH, 'r') as f:
    feature_names = json.load(f)

def prepare_input(data):
    """Prepare input data for prediction"""
    # Create a DataFrame with all features set to 0
    input_df = pd.DataFrame(columns=feature_names)
    
    # Update with provided values
    for feature in feature_names:
        if feature in data:
            input_df[feature] = [float(data[feature])]
        else:
            input_df[feature] = [0.0]  # Default value
    
    # Ensure correct column order
    input_df = input_df[feature_names]
    
    return input_df

@app.route('/')
def home():
    """Serve the main web page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get data from request
        data = request.json
        
        # Prepare input
        input_df = prepare_input(data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Decode prediction
        risk_level = encoder.inverse_transform([prediction])[0]
        
        # Create response
        response = {
            'success': True,
            'prediction': {
                'risk_level': risk_level,
                'confidence': float(np.max(probabilities)),
                'probabilities': {
                    encoder.classes_[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            },
            'input_features': data,
            'model_info': {
                'type': 'Decision Tree',
                'accuracy': 0.9766,  # Update with your actual accuracy
                'features_used': feature_names
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    # Load feature importance
    feature_importance = []
    if os.path.exists('feature_importance.csv'):
        df = pd.read_csv('feature_importance.csv')
        feature_importance = df.to_dict('records')
    
    info = {
        'model_name': 'Metro Manila Pollution Risk Classifier',
        'accuracy': 0.9766,  # Your actual accuracy
        'dataset': 'January-November 2025 Air Quality Data',
        'features': feature_names,
        'target_classes': encoder.classes_.tolist(),
        'feature_importance': feature_importance,
        'risk_thresholds': {
            'low': 'PM2.5 ≤ 12 μg/m³',
            'moderate': '12 < PM2.5 ≤ 35 μg/m³',
            'high': 'PM2.5 > 35 μg/m³'
        }
    }
    
    return jsonify(info)

@app.route('/api/sample_predictions', methods=['GET'])
def sample_predictions():
    """Get sample predictions for demonstration"""
    samples = [
        {
            'name': 'Clean Day',
            'inputs': {
                'PM2.5': 8.5, 'PM10': 25.0, 'NO2': 15.0,
                'SO2': 5.0, 'CO': 0.8, 'O3': 35.0,
                'Temperature': 28.5, 'Humidity': 65.0
            }
        },
        {
            'name': 'Moderate Pollution',
            'inputs': {
                'PM2.5': 25.0, 'PM10': 55.0, 'NO2': 35.0,
                'SO2': 12.0, 'CO': 2.0, 'O3': 55.0,
                'Temperature': 30.0, 'Humidity': 75.0
            }
        },
        {
            'name': 'High Pollution',
            'inputs': {
                'PM2.5': 45.0, 'PM10': 85.0, 'NO2': 65.0,
                'SO2': 25.0, 'CO': 4.5, 'O3': 75.0,
                'Temperature': 32.0, 'Humidity': 85.0
            }
        }
    ]
    
    return jsonify(samples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
