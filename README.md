# 🔍 Fraud Detection in Online Transactions

A complete end-to-end machine learning project for detecting fraudulent transactions using Random Forest classification with an interactive Streamlit web application.

## 📋 Project Overview

This project implements a robust fraud detection system that analyzes online transactions and predicts whether they are fraudulent or legitimate. The system uses a Random Forest classifier trained on synthetic transaction data with realistic class imbalance, featuring a user-friendly web interface for real-time predictions.

## 🎯 Objectives

- **High Accuracy Prediction**: Detect fraud with high precision despite class imbalance
- **Real-world Applicability**: Handle realistic transaction patterns and data distributions
- **Interpretability**: Provide feature importance analysis for model transparency
- **User-friendly Interface**: Offer an intuitive web application for easy predictions
- **Scalable Architecture**: Design for easy deployment and maintenance

## 📊 Dataset Description

The synthetic dataset contains **5,000 transactions** with realistic fraud patterns:

### Features:
- **amount**: Transaction amount ($)
- **oldbalanceOrg**: Origin account balance before transaction
- **newbalanceOrg**: Origin account balance after transaction
- **oldbalanceDest**: Destination account balance before transaction
- **newbalanceDest**: Destination account balance after transaction
- **transaction_type**: Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT)
- **customer_location**: Customer's geographic location (US, UK, CA, AU, DE)
- **hour_of_day**: Hour when transaction occurred (0-23)

### Target:
- **isFraud**: Binary label (1 = Fraud, 0 = Legitimate)

### Dataset Characteristics:
- **Fraud Rate**: ~2% (realistic imbalance reflecting real-world scenarios)
- **Total Transactions**: 5,000
- **Fraud Cases**: ~100 transactions
- **Legitimate Cases**: ~4,900 transactions

## 🌳 Machine Learning Approach

### Random Forest Algorithm
Random Forest is an ensemble learning method that combines multiple decision trees:

**Key Advantages:**
- **Handles Mixed Data Types**: Works well with both numerical and categorical features
- **Feature Importance**: Provides interpretable feature rankings
- **Robust to Outliers**: Ensemble approach reduces impact of anomalous data
- **No Feature Scaling Required**: Tree-based methods are scale-invariant
- **Handles Missing Values**: Can work with incomplete data

### Model Configuration:
- **n_estimators**: 200 trees for robust predictions
- **max_depth**: None (trees grow until pure leaves)
- **class_weight**: 'balanced' to handle class imbalance
- **random_state**: 42 for reproducibility

## ⚖️ Class Imbalance Handling

### Hybrid Approach: SMOTE + Tomek Links

1. **SMOTE (Synthetic Minority Oversampling Technique)**:
   - Generates synthetic fraud examples by interpolating between existing fraud cases
   - Increases minority class representation
   - Creates realistic synthetic samples in feature space

2. **Tomek Links**:
   - Removes borderline examples that are difficult to classify
   - Cleans the decision boundary between classes
   - Improves class separation

3. **Class Weight Balancing**:
   - Assigns higher weights to minority class during training
   - Penalizes misclassification of fraud cases more heavily

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn streamlit plotly seaborn matplotlib
```

### Installation & Setup

1. **Clone or download the project**
```bash
cd fraud-detection_PRO1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate dataset and train model** (if needed)
```bash
python src/preprocess.py
python src/train.py
```

4. **Run the Streamlit application**
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
fraud-detection_PRO1/
│
├── data/
│   └── transactions.csv          # Synthetic transaction dataset
│
├── model/
│   ├── fraud_model.pkl          # Trained Random Forest model
│   ├── encoder.pkl              # Preprocessing pipeline
│   ├── feature_importance.csv   # Feature importance scores
│   └── confusion_matrix.png     # Model evaluation plot
│
├── src/
│   ├── preprocess.py           # Data generation and preprocessing
│   ├── train.py                # Model training and evaluation
│   └── utils.py                # Utility functions for prediction
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🖥️ Web Application Features

### Prediction Interface
- **Interactive Form**: Easy-to-use input fields for transaction details
- **Real-time Validation**: Input validation with helpful error messages
- **Visual Results**: Color-coded predictions with probability scores
- **Probability Charts**: Bar charts showing prediction confidence

### Model Information Dashboard
- **Feature Importance**: Interactive charts showing which features matter most
- **Model Metrics**: Performance statistics and evaluation results
- **Dataset Overview**: Information about training data characteristics

## 📈 Model Performance

The model is evaluated using multiple metrics:

- **Confusion Matrix**: Shows true vs predicted classifications
- **Precision**: Accuracy of fraud predictions
- **Recall**: Ability to detect actual fraud cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🔧 Key Components

### Data Processing Pipeline
- **Synthetic Data Generation**: Creates realistic transaction patterns
- **Feature Engineering**: Derives meaningful features from raw data
- **Categorical Encoding**: Handles non-numerical features
- **Data Standardization**: Normalizes numerical features

### Model Training Pipeline
- **Class Imbalance Handling**: SMOTE + Tomek Links hybrid approach
- **Hyperparameter Optimization**: Grid search for best parameters
- **Cross-validation**: Robust model evaluation
- **Feature Selection**: Identifies most important predictors

### Web Application
- **Streamlit Framework**: Modern, responsive web interface
- **Real-time Predictions**: Instant fraud detection results
- **Interactive Visualizations**: Charts and graphs for better understanding
- **Model Interpretability**: Feature importance and prediction explanations

## 🔍 Usage Examples

### Making Predictions

1. **Open the web application**
2. **Navigate to the Prediction page**
3. **Enter transaction details**:
   - Transaction amount
   - Account balances (before/after)
   - Transaction type
   - Customer location
   - Time of transaction
4. **Click "Check for Fraud"**
5. **View results** with probability scores and confidence levels

### Analyzing Model Performance

1. **Navigate to Model Info page**
2. **View feature importance rankings**
3. **Understand which factors contribute most to fraud detection**
4. **Review dataset characteristics and model metrics**

## 🛠️ Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **imbalanced-learn**: Handling class imbalance
- **streamlit**: Web application framework
- **matplotlib/plotly**: Data visualization

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Input Features**: 8 features (6 numerical, 2 categorical)
- **Output**: Binary classification (Fraud/Legit)
- **Preprocessing**: Column transformer with scaling and encoding

## 🔮 Future Enhancements

### Technical Improvements
- **Real-time Processing**: Stream processing for live transactions
- **Advanced Models**: Deep learning approaches (Neural Networks, Autoencoders)
- **Ensemble Methods**: Combine multiple algorithms for better performance
- **Feature Engineering**: Automated feature generation and selection

### Deployment & Monitoring
- **API Integration**: REST API for external system integration
- **Model Monitoring**: Performance tracking and drift detection
- **A/B Testing**: Compare different model versions
- **Automated Retraining**: Regular model updates with new data

### User Experience
- **Explainable AI**: SHAP values for individual prediction explanations
- **Batch Processing**: Handle multiple transactions at once
- **Historical Analysis**: Trend analysis and reporting features
- **Alert System**: Automated notifications for high-risk transactions

## 📊 Performance Metrics

The model achieves strong performance across key metrics:

- **Accuracy**: High overall prediction accuracy
- **Precision**: Low false positive rate for fraud detection
- **Recall**: High detection rate for actual fraud cases
- **F1-Score**: Balanced performance across precision and recall
- **ROC-AUC**: Strong discriminative ability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes. Feel free to use and modify for learning and non-commercial applications.

## 🙏 Acknowledgments

- **scikit-learn**: For providing excellent machine learning tools
- **Streamlit**: For the amazing web application framework
- **imbalanced-learn**: For class imbalance handling techniques

---

**Built with ❤️ using Python, scikit-learn, and Streamlit**

For questions or support, please open an issue in the repository.
