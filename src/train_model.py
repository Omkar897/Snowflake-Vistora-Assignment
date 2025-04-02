import sys
import os

# Ensure the parent directory is in the Python path
from utils.snowflake_utils import execute_query
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Get the directory where the script is located
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'models')

def check_existing_models():
    """Check if model files exist and warn about overwriting"""
    model_path = os.path.join(MODELS_DIR, 'customer_spending_model.joblib')
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessors.joblib')
    
    if os.path.exists(model_path) or os.path.exists(preprocessor_path):
        print("\nWARNING: Existing model files will be overwritten!")
        print("This will affect all predictions using the current model.")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return False
    return True

def load_features():
    """Load features from Snowflake Feature Store"""
    query = """
    SELECT *
    FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES
    """
    return execute_query(query)

def prepare_features(df):
    """Prepare features for model training"""
    feature_columns = [
        'TOTAL_ORDERS', 'AVG_ORDER_VALUE', 'CUSTOMER_LIFETIME_DAYS',
        'TOTAL_ITEMS_ORDERED', 'AVG_QUANTITY_PER_ITEM', 'AVG_DISCOUNT',
        'AVG_DAILY_SPEND', 'ORDERS_PER_MONTH', 'ITEMS_PER_ORDER',
        'MARKET_SEGMENT', 'NATION', 'SPENDING_SEGMENT', 'FREQUENCY_SEGMENT'
    ]
    
    X = df[feature_columns].copy()
    y = df['AVG_DAILY_SPEND'] * 30  # Monthly spending prediction
    
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    numerical_columns = [
        'TOTAL_ORDERS', 'AVG_ORDER_VALUE', 'CUSTOMER_LIFETIME_DAYS',
        'TOTAL_ITEMS_ORDERED', 'AVG_QUANTITY_PER_ITEM', 'AVG_DISCOUNT',
        'AVG_DAILY_SPEND', 'ORDERS_PER_MONTH', 'ITEMS_PER_ORDER'
    ]
    categorical_columns = ['MARKET_SEGMENT', 'NATION', 'SPENDING_SEGMENT', 'FREQUENCY_SEGMENT']
    
    numerical_imputer = SimpleImputer(strategy='mean')
    X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])
    
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='UNKNOWN')
    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
    
    encoders = {}
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])
    
    joblib.dump({
        'numerical_imputer': numerical_imputer,
        'categorical_imputer': categorical_imputer,
        'encoders': encoders
    }, os.path.join(MODELS_DIR, 'preprocessors.joblib'))
    
    return X, y

def train_model():
    if not check_existing_models():
        return None, None
        
    print("Loading features from Feature Store...")
    df = load_features()
    
    print("\nPreparing features for training...")
    X, y = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance Metrics:")
    print(f"Root Mean Square Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    print("\nSaving model and artifacts...")
    model_data = {
        'model': model,
        'feature_names': X.columns.tolist()
    }
    joblib.dump(model_data, os.path.join(MODELS_DIR, 'customer_spending_model.joblib'))
    
    print("\nSample Prediction:")
    sample_customer = pd.DataFrame([X_test.iloc[0]], columns=X.columns)
    predicted_spending = model.predict(sample_customer)[0]
    print(f"Predicted Monthly Spending: ${predicted_spending:.2f}")
    
    return model, feature_importance

if __name__ == "__main__":
    model, feature_importance = train_model()
