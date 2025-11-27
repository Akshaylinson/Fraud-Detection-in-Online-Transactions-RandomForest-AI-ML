import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

def generate_synthetic_data():
    """Generate synthetic fraud detection dataset"""
    np.random.seed(42)
    
    # Generate 5000 transactions
    n_samples = 5000
    
    # Transaction types
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']
    
    # Generate base features
    data = {
        'transaction_id': [f'T{i:06d}' for i in range(1, n_samples + 1)],
        'amount': np.random.lognormal(mean=5, sigma=2, size=n_samples),
        'oldbalanceOrg': np.random.lognormal(mean=8, sigma=1.5, size=n_samples),
        'newbalanceOrg': np.zeros(n_samples),
        'oldbalanceDest': np.random.lognormal(mean=7, sigma=1.8, size=n_samples),
        'newbalanceDest': np.zeros(n_samples),
        'transaction_type': np.random.choice(transaction_types, size=n_samples),
        'device_id': [f'D{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'customer_location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], size=n_samples),
        'hour_of_day': np.random.randint(0, 24, size=n_samples)
    }
    
    # Calculate new balances
    data['newbalanceOrg'] = data['oldbalanceOrg'] - data['amount']
    data['newbalanceDest'] = data['oldbalanceDest'] + data['amount']
    
    # Generate fraud labels (1-3% fraud rate)
    fraud_rate = 0.02
    n_fraud = int(n_samples * fraud_rate)
    fraud_labels = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    fraud_labels[fraud_indices] = 1
    
    # Make fraud transactions more suspicious
    for idx in fraud_indices:
        # Higher amounts for fraud
        data['amount'][idx] *= np.random.uniform(2, 5)
        # Suspicious balance changes
        if np.random.random() > 0.5:
            data['newbalanceOrg'][idx] = data['oldbalanceOrg'][idx]  # No balance change
        # Prefer certain transaction types for fraud
        if np.random.random() > 0.3:
            data['transaction_type'][idx] = np.random.choice(['TRANSFER', 'CASH_OUT'])
    
    data['isFraud'] = fraud_labels
    
    df = pd.DataFrame(data)
    return df

def preprocess_data(df, fit_encoder=True, encoder=None):
    """Preprocess the dataset"""
    # Handle missing values
    df = df.fillna(0)
    
    # Define feature columns
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 
                         'oldbalanceDest', 'newbalanceDest', 'hour_of_day']
    categorical_features = ['transaction_type', 'customer_location']
    
    # Create preprocessor
    if fit_encoder:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])
        
        # Fit and transform
        X_features = df[numerical_features + categorical_features]
        X_processed = preprocessor.fit_transform(X_features)
        
        return X_processed, preprocessor
    else:
        # Use existing encoder
        X_features = df[numerical_features + categorical_features]
        X_processed = encoder.transform(X_features)
        return X_processed

def create_dataset():
    """Create and save the synthetic dataset"""
    print("Generating synthetic fraud detection dataset...")
    df = generate_synthetic_data()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    df.to_csv('data/transactions.csv', index=False)
    print(f"Dataset saved with {len(df)} transactions ({df['isFraud'].sum()} fraud cases)")
    
    return df

def load_and_split_data():
    """Load dataset and create train/test split"""
    # Create dataset if it doesn't exist
    if not os.path.exists('data/transactions.csv'):
        df = create_dataset()
    else:
        df = pd.read_csv('data/transactions.csv')
    
    # Remove transaction_id and device_id for modeling
    features_df = df.drop(['transaction_id', 'device_id', 'isFraud'], axis=1)
    target = df['isFraud']
    
    # Preprocess features
    X_processed, encoder = preprocess_data(features_df, fit_encoder=True)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # Save encoder
    os.makedirs('model', exist_ok=True)
    with open('model/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    create_dataset()
