import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import predict_fraud, get_feature_importance, validate_input_data

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Fraud Detection in Online Transactions")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Info"])
    
    if page == "Prediction":
        prediction_page()
    else:
        model_info_page()

def prediction_page():
    st.header("Transaction Fraud Detection")
    st.markdown("Enter transaction details to check for potential fraud")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Amount of the transaction"
        )
        
        oldbalanceOrg = st.number_input(
            "Original Balance - Origin ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Balance before transaction at origin account"
        )
        
        newbalanceOrg = st.number_input(
            "New Balance - Origin ($)",
            min_value=0.0,
            value=900.0,
            step=100.0,
            help="Balance after transaction at origin account"
        )
        
        oldbalanceDest = st.number_input(
            "Original Balance - Destination ($)",
            min_value=0.0,
            value=500.0,
            step=100.0,
            help="Balance before transaction at destination account"
        )
    
    with col2:
        st.subheader("Additional Information")
        
        newbalanceDest = st.number_input(
            "New Balance - Destination ($)",
            min_value=0.0,
            value=600.0,
            step=100.0,
            help="Balance after transaction at destination account"
        )
        
        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
            help="Type of transaction"
        )
        
        customer_location = st.selectbox(
            "Customer Location",
            ["US", "UK", "CA", "AU", "DE"],
            help="Customer's location"
        )
        
        hour_of_day = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour when transaction occurred (0-23)"
        )
    
    # Prediction button
    if st.button("🔍 Check for Fraud", type="primary"):
        try:
            # Prepare input data
            input_data = {
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrg': newbalanceOrg,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'transaction_type': transaction_type,
                'customer_location': customer_location,
                'hour_of_day': hour_of_day
            }
            
            # Validate input
            validate_input_data(input_data)
            
            # Make prediction
            result = predict_fraud(input_data)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if result['prediction'] == 1:
                    st.error("🔴 FRAUD DETECTED")
                    st.markdown(f"**Fraud Probability:** {result['probability_fraud']:.2%}")
                else:
                    st.success("🟢 LEGIT TRANSACTION")
                    st.markdown(f"**Legit Probability:** {result['probability_legit']:.2%}")
                
                st.markdown(f"**Confidence:** {result['confidence']:.2%}")
            
            with result_col2:
                # Probability bar chart
                fig, ax = plt.subplots(figsize=(6, 4))
                categories = ['Legit', 'Fraud']
                probabilities = [result['probability_legit'], result['probability_fraud']]
                colors = ['green', 'red']
                
                bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.2%}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def model_info_page():
    st.header("Model Information")
    
    # Model description
    st.subheader("About the Model")
    st.markdown("""
    This fraud detection system uses a **Random Forest Classifier** trained on synthetic transaction data.
    
    **Key Features:**
    - **Algorithm:** Random Forest with 200 estimators
    - **Class Imbalance Handling:** SMOTE + Tomek Links hybrid approach
    - **Features:** Transaction amounts, account balances, transaction type, location, and timing
    - **Target:** Binary classification (Fraud vs Legit)
    """)
    
    # Feature importance
    try:
        st.subheader("Feature Importance")
        importance_data = get_feature_importance()
        
        if importance_data:
            # Convert to DataFrame for plotting
            df_importance = pd.DataFrame(importance_data)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = df_importance.head(10)
            
            bars = ax.barh(top_features['feature'], top_features['importance'])
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            ax.set_title('Top 10 Most Important Features')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show table
            st.subheader("Feature Importance Table")
            st.dataframe(df_importance, use_container_width=True)
        else:
            st.warning("Feature importance data not available. Please train the model first.")
    
    except Exception as e:
        st.error(f"Error loading feature importance: {str(e)}")
    
    # Dataset info
    st.subheader("Dataset Information")
    st.markdown("""
    **Dataset Characteristics:**
    - **Size:** 5,000 transactions
    - **Fraud Rate:** ~2% (realistic imbalance)
    - **Features:** 9 input features
    - **Split:** 80% training, 20% testing
    
    **Features Used:**
    1. **amount** - Transaction amount
    2. **oldbalanceOrg** - Origin account balance before transaction
    3. **newbalanceOrg** - Origin account balance after transaction
    4. **oldbalanceDest** - Destination account balance before transaction
    5. **newbalanceDest** - Destination account balance after transaction
    6. **transaction_type** - Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT)
    7. **customer_location** - Customer's geographic location
    8. **hour_of_day** - Hour when transaction occurred
    """)

if __name__ == "__main__":
    main()
