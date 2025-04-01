import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.snowflake_utils import execute_query
import joblib
import pandas as pd

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'models')

def load_customer_data(customer_id):
    """Load data for a specific customer from Snowflake"""
    query = f"""
    SELECT 
        TOTAL_ORDERS, AVG_ORDER_VALUE, CUSTOMER_LIFETIME_DAYS,
        TOTAL_ITEMS_ORDERED, AVG_QUANTITY_PER_ITEM, AVG_DISCOUNT,
        AVG_DAILY_SPEND, ORDERS_PER_MONTH, ITEMS_PER_ORDER,
        MARKET_SEGMENT, NATION, SPENDING_SEGMENT, FREQUENCY_SEGMENT
    FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES
    WHERE CUSTOMER_ID = '{customer_id}'
    """
    return execute_query(query)

def prepare_customer_data(customer_df):
    """Prepare customer data for prediction using saved preprocessors"""
    # Load preprocessors
    preprocessors = joblib.load(os.path.join(MODELS_DIR, 'preprocessors.joblib'))
    numerical_imputer = preprocessors['numerical_imputer']
    categorical_imputer = preprocessors['categorical_imputer']
    encoders = preprocessors['encoders']
    
    # Separate numerical and categorical columns
    numerical_columns = [
        'TOTAL_ORDERS', 'AVG_ORDER_VALUE', 'CUSTOMER_LIFETIME_DAYS',
        'TOTAL_ITEMS_ORDERED', 'AVG_QUANTITY_PER_ITEM', 'AVG_DISCOUNT',
        'AVG_DAILY_SPEND', 'ORDERS_PER_MONTH', 'ITEMS_PER_ORDER'
    ]
    categorical_columns = ['MARKET_SEGMENT', 'NATION', 'SPENDING_SEGMENT', 'FREQUENCY_SEGMENT']
    
    # Handle missing values
    customer_df[numerical_columns] = numerical_imputer.transform(customer_df[numerical_columns])
    customer_df[categorical_columns] = categorical_imputer.transform(customer_df[categorical_columns])
    
    # Encode categorical variables
    for col in categorical_columns:
        customer_df[col] = encoders[col].transform(customer_df[col])
    
    return customer_df

def predict_spending(customer_id):
    """Predict monthly spending for a specific customer"""
    # Load customer data
    print(f"\nLoading data for customer {customer_id}...")
    customer_df = load_customer_data(customer_id)
    
    if customer_df.empty:
        print(f"No data found for customer {customer_id}")
        return None
    
    # Prepare customer data
    print("Preparing customer data...")
    customer_features = prepare_customer_data(customer_df)
    
    # Load model
    print("Loading model...")
    model_data = joblib.load(os.path.join(MODELS_DIR, 'customer_spending_model.joblib'))
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Ensure customer features have the same column order as training data
    customer_features = customer_features[feature_names]
    
    # Make prediction
    predicted_spending = model.predict(customer_features)[0]
    current_monthly_spend = customer_df['AVG_DAILY_SPEND'].iloc[0] * 30
    
    print("\nPrediction Results:")
    print(f"Customer ID: {customer_id}")
    print(f"Current Monthly Spending: ${current_monthly_spend:.2f}")
    print(f"Predicted Next Month Spending: ${predicted_spending:.2f}")
    print(f"Expected Change: ${(predicted_spending - current_monthly_spend):.2f}")
    
    # Print customer segments
    print("\nCustomer Segments:")
    print(f"Market Segment: {customer_df['MARKET_SEGMENT'].iloc[0]}")
    print(f"Nation: {customer_df['NATION'].iloc[0]}")
    print(f"Spending Segment: {customer_df['SPENDING_SEGMENT'].iloc[0]}")
    print(f"Frequency Segment: {customer_df['FREQUENCY_SEGMENT'].iloc[0]}")
    
    return predicted_spending

def main():
    while True:
        customer_id = input("\nEnter customer ID (or 'q' to quit): ")
        if customer_id.lower() == 'q':
            break
        predict_spending(customer_id)

if __name__ == "__main__":
    main() 