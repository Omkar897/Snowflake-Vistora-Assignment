import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load environment variables
load_dotenv()

def get_snowflake_connection():
    """Create and return a Snowflake connection"""
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )

def execute_query(query):
    """Execute a query and return results as a pandas DataFrame"""
    conn = get_snowflake_connection()
    try:
        cur = conn.cursor()
        cur.execute(query)
        results = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(results, columns=columns)
    finally:
        cur.close()
        conn.close()

def create_feature_table(table_name, features_df):
    """Store features in Snowflake"""
    conn = get_snowflake_connection()
    try:
        # Convert DataFrame to Snowflake-compatible format
        features_df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Features stored in table: {table_name}")
    finally:
        conn.close()

def get_features(table_name):
    """Retrieve features from Snowflake"""
    query = f"SELECT * FROM {table_name}"
    return execute_query(query)

def engineer_features(raw_data):
    """Perform feature engineering on the raw data"""
    # Example feature engineering steps
    # Replace these with your actual feature engineering logic
    features = raw_data.copy()
    
    # Add derived features
    if 'amount' in features.columns:
        features['amount_squared'] = features['amount'] ** 2
        features['amount_log'] = features['amount'].apply(lambda x: np.log1p(x))
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = features.select_dtypes(include=['float64', 'int64']).columns
    features[numerical_columns] = scaler.fit_transform(features[numerical_columns])
    
    return features 