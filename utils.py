import pickle
import pandas as pd
import numpy as np
import os

def load_model_and_encoder():
    """Load the trained model and encoder"""
    try:
        # Load model
        with open('model/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoder
        with open('model/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        return model, encoder
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found. Please run training first. Error: {e}")

def preprocess_input_data(input_data, encoder):
    """Preprocess input data for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Define feature columns (same order as training)
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 
                         'oldbalanceDest', 'newbalanceDest', 'hour_of_day']
    categorical_features = ['transaction_type', 'customer_location']
    
    # Select and order features
    features_df = df[numerical_features + categorical_features]
    
    # Transform using the fitted encoder
    X_processed = encoder.transform(features_df)
    
    return X_processed

def predict_fraud(input_data):
    """Make fraud prediction for input data"""
    try:
        # Load model and encoder
        model, encoder = load_model_and_encoder()
        
        # Preprocess input
        X_processed = preprocess_input_data(input_data, encoder)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        return {
            'prediction': int(prediction),
            'probability_legit': float(probability[0]),
            'probability_fraud': float(probability[1]),
            'confidence': float(max(probability))
        }
    
    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

def get_feature_importance():
    """Get feature importance from the trained model"""
    try:
        # Load model
        with open('model/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load feature importance if available
        if os.path.exists('model/feature_importance.csv'):
            importance_df = pd.read_csv('model/feature_importance.csv')
            return importance_df.to_dict('records')
        else:
            # Create basic feature importance
            feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 
                           'newbalanceDest', 'hour_of_day', 'transaction_type', 'customer_location']
            
            # Pad or truncate to match actual features
            n_features = len(model.feature_importances_)
            if len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
            elif len(feature_names) < n_features:
                feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
            
            importance_data = []
            for i, importance in enumerate(model.feature_importances_):
                importance_data.append({
                    'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                    'importance': float(importance)
                })
            
            return sorted(importance_data, key=lambda x: x['importance'], reverse=True)
    
    except Exception as e:
        raise Exception(f"Failed to get feature importance: {e}")

def validate_input_data(input_data):
    """Validate input data format and values"""
    required_fields = [
        'amount', 'oldbalanceOrg', 'newbalanceOrg', 
        'oldbalanceDest', 'newbalanceDest', 'transaction_type',
        'customer_location', 'hour_of_day'
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate data types and ranges
    try:
        # Numerical validations
        amount = float(input_data['amount'])
        if amount < 0:
            raise ValueError("Amount must be non-negative")
        
        oldbalanceOrg = float(input_data['oldbalanceOrg'])
        newbalanceOrg = float(input_data['newbalanceOrg'])
        oldbalanceDest = float(input_data['oldbalanceDest'])
        newbalanceDest = float(input_data['newbalanceDest'])
        
        hour_of_day = int(input_data['hour_of_day'])
        if not (0 <= hour_of_day <= 23):
            raise ValueError("Hour of day must be between 0 and 23")
        
        # Categorical validations
        valid_transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']
        if input_data['transaction_type'] not in valid_transaction_types:
            raise ValueError(f"Transaction type must be one of: {valid_transaction_types}")
        
        valid_locations = ['US', 'UK', 'CA', 'AU', 'DE']
        if input_data['customer_location'] not in valid_locations:
            raise ValueError(f"Customer location must be one of: {valid_locations}")
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input data: {e}")
    
    return True
