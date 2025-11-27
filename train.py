import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_split_data

def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE + Tomek Links hybrid approach"""
    print("Original class distribution:")
    print(f"Class 0 (Legit): {sum(y_train == 0)}")
    print(f"Class 1 (Fraud): {sum(y_train == 1)}")
    
    # Apply SMOTE + Tomek Links
    smote_tomek = SMOTETomek(
        smote=SMOTE(random_state=42),
        tomek=TomekLinks(),
        random_state=42
    )
    
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    print("\nAfter SMOTE + Tomek Links:")
    print(f"Class 0 (Legit): {sum(y_resampled == 0)}")
    print(f"Class 1 (Fraud): {sum(y_resampled == 1)}")
    
    return X_resampled, y_resampled

def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest Classifier...")
    
    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X_train, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_resampled, y_resampled)
    
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_model(model):
    """Save the trained model"""
    with open('model/fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to model/fraud_model.pkl")

def main():
    """Main training pipeline"""
    print("Starting fraud detection model training...")
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model)
    
    # Feature importance
    feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 
                    'newbalanceDest', 'hour_of_day', 'transaction_type_DEBIT',
                    'transaction_type_PAYMENT', 'transaction_type_TRANSFER',
                    'customer_location_CA', 'customer_location_DE', 
                    'customer_location_UK', 'customer_location_US']
    
    # Adjust feature names based on actual encoder output
    n_features = len(model.feature_importances_)
    if len(feature_names) != n_features:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:n_features],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10))
    
    # Save feature importance
    importance_df.to_csv('model/feature_importance.csv', index=False)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
