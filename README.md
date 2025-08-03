# 💳 Fraud Transaction Detection Project

A machine learning-based system for detecting fraudulent credit card transactions using Random Forest classification. This project includes both a web application for real-time predictions and batch processing capabilities.

## 🎯 Project Overview

This project demonstrates an end-to-end machine learning pipeline for fraud detection in credit card transactions. It processes transaction data, trains a Random Forest model, and provides both interactive and batch prediction capabilities through a Streamlit web application.

## ✨ Features

- **Real-time Fraud Detection**: Interactive web interface for manual transaction input
- **Batch Processing**: Upload CSV files for bulk transaction analysis
- **Machine Learning Model**: Random Forest classifier trained on 1.3M+ transactions
- **Feature Engineering**: Age calculation and categorical encoding
- **Probability Scoring**: Provides fraud probability scores for each transaction
- **Configurable Threshold**: Adjustable fraud detection threshold (default: 0.28)

## 🏗️ Project Structure

```
Fraud Transaction Detection Project/
├── data/                          # Dataset files
│   ├── credit_card_transactions.csv    # Main dataset (338MB)
│   ├── fraudulent_transactions.csv     # Fraudulent transactions subset
│   └── sample_transactions.csv         # Sample data for testing
├── models/                        # Trained models
│   └── rf_fraud_model.pkl        # Random Forest model (62MB)
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda_preprocessing.ipynb     # Exploratory data analysis
│   └── 02_train_and_save_model.ipynb  # Model training pipeline
├── src/                          # Application source code
│   └── app.py                    # Streamlit web application
└── README.md                     # This file
```

## 📊 Dataset

The project uses a comprehensive credit card transaction dataset with the following features:

- **Transaction Details**: Amount, merchant, category, date/time
- **Customer Information**: Gender, job, age, location
- **Geographic Data**: Customer and merchant coordinates, city population
- **Target Variable**: Fraud flag (0 = legitimate, 1 = fraudulent)

**Dataset Statistics:**

- Total transactions: 1,296,675
- Features: 23 columns
- Fraud rate: Imbalanced dataset (fraudulent transactions are rare)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "Fraud Transaction Detection Project"
   ```

2. **Install dependencies**

   ```bash
   pip install streamlit pandas scikit-learn joblib numpy matplotlib
   ```

3. **Download the dataset and model**
   - Place `credit_card_transactions.csv` in the `data/` directory
   - Place `rf_fraud_model.pkl` in the `models/` directory

### Running the Application

1. **Start the Streamlit app**

   ```bash
   streamlit run src/app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The application will load with the fraud detection interface

## 🎮 Usage

### Manual Transaction Input

1. Use the sidebar to input transaction details:

   - **Merchant**: Select from available merchants
   - **Category**: Choose transaction category
   - **Amount**: Enter transaction amount
   - **Customer Info**: Gender, job, state, age
   - **Location**: City population, merchant coordinates

2. View results:
   - **Fraud Prediction**: Legitimate (✅) or Fraudulent (🚨)
   - **Fraud Probability**: Percentage score indicating risk level

### Batch Processing

1. **Prepare CSV file** with the same columns as the training dataset
2. **Upload file** using the file uploader
3. **View results** in a table format with:
   - Original transaction data
   - Fraud probability scores
   - Fraud flags based on threshold

## 🔧 Technical Details

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Features**: 10 engineered features
  - 5 categorical (encoded): merchant, category, gender, job, state
  - 5 numerical: amount, city_pop, merch_lat, merch_long, age
- **Threshold**: 0.28 (configurable)
- **Performance**: Optimized for fraud detection in imbalanced datasets

### Feature Engineering

- **Age Calculation**: Derived from transaction date and date of birth
- **Categorical Encoding**: Integer encoding with unseen category handling
- **Data Preprocessing**: Missing value handling and data type conversion

### Model Training Pipeline

1. **Data Loading**: Load and clean transaction data
2. **Feature Engineering**: Calculate age and encode categorical variables
3. **Train-Test Split**: 80/20 split with class balance preservation
4. **Model Training**: Random Forest with optimized hyperparameters
5. **Model Persistence**: Save trained model using joblib

## 📈 Performance Metrics

The model is evaluated using:

- **Classification Report**: Precision, recall, F1-score
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/false positive/negative rates

## 🔍 Exploratory Data Analysis

The `01_eda_preprocessing.ipynb` notebook includes:

- Data shape and type analysis
- Missing value assessment
- Statistical summaries
- Distribution visualizations
- Feature correlation analysis

## 🛠️ Development

### Adding New Features

1. **Modify feature engineering** in the training notebook
2. **Update the Streamlit app** to include new input fields
3. **Retrain the model** with new features
4. **Test the application** with sample data

### Model Improvements

- Experiment with different algorithms (XGBoost, LightGBM)
- Implement feature selection techniques
- Add ensemble methods
- Optimize hyperparameters using grid search

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue in the repository.

---

**Note**: This project uses a large dataset (338MB) and trained model (62MB). Ensure you have sufficient storage space and download the required files before running the application.
