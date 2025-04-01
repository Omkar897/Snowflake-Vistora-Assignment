import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.snowflake_utils import execute_query
import pandas as pd
import numpy as np

def engineer_features(raw_data):
    """Perform feature engineering on the raw data"""
    features = raw_data.copy()
    
    # Calculate basic metrics
    features['TOTAL_ORDERS'] = features.groupby('CUSTOMER_ID')['ORDER_ID'].nunique()
    features['AVG_ORDER_VALUE'] = features.groupby('CUSTOMER_ID')['ORDER_TOTAL'].mean()
    
    # Calculate customer lifetime
    features['CUSTOMER_LIFETIME_DAYS'] = (
        features.groupby('CUSTOMER_ID')['ORDER_DATE'].max() - 
        features.groupby('CUSTOMER_ID')['ORDER_DATE'].min()
    ).dt.days
    
    # Calculate item metrics
    features['TOTAL_ITEMS_ORDERED'] = features.groupby('CUSTOMER_ID')['QUANTITY'].sum()
    features['AVG_QUANTITY_PER_ITEM'] = features.groupby('CUSTOMER_ID')['QUANTITY'].mean()
    
    # Calculate spending metrics
    features['AVG_DISCOUNT'] = features.groupby('CUSTOMER_ID')['DISCOUNT'].mean()
    features['AVG_DAILY_SPEND'] = features.groupby('CUSTOMER_ID')['ORDER_TOTAL'].sum() / features['CUSTOMER_LIFETIME_DAYS']
    
    # Calculate order frequency metrics
    features['ORDERS_PER_MONTH'] = features['TOTAL_ORDERS'] / (features['CUSTOMER_LIFETIME_DAYS'] / 30)
    features['ITEMS_PER_ORDER'] = features['TOTAL_ITEMS_ORDERED'] / features['TOTAL_ORDERS']
    
    # Create customer segments
    features['SPENDING_SEGMENT'] = pd.qcut(
        features['AVG_DAILY_SPEND'],
        q=3,
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    features['FREQUENCY_SEGMENT'] = pd.qcut(
        features['ORDERS_PER_MONTH'],
        q=3,
        labels=['OCCASIONAL', 'REGULAR', 'FREQUENT']
    )
    
    # Keep only the engineered features
    feature_columns = [
        'CUSTOMER_ID', 'TOTAL_ORDERS', 'AVG_ORDER_VALUE', 'CUSTOMER_LIFETIME_DAYS',
        'TOTAL_ITEMS_ORDERED', 'AVG_QUANTITY_PER_ITEM', 'AVG_DISCOUNT',
        'AVG_DAILY_SPEND', 'ORDERS_PER_MONTH', 'ITEMS_PER_ORDER',
        'MARKET_SEGMENT', 'NATION', 'SPENDING_SEGMENT', 'FREQUENCY_SEGMENT'
    ]
    
    return features[feature_columns].drop_duplicates()

def load_raw_data():
    """Load raw data from Snowflake"""
    query = """
    SELECT 
        c.C_CUSTKEY as CUSTOMER_ID,
        o.O_ORDERKEY as ORDER_ID,
        o.O_ORDERDATE as ORDER_DATE,
        l.L_EXTENDEDPRICE * (1 - l.L_DISCOUNT) as ORDER_TOTAL,
        l.L_QUANTITY as QUANTITY,
        l.L_DISCOUNT as DISCOUNT,
        c.C_MKTSEGMENT as MARKET_SEGMENT,
        n.N_NAME as NATION
    FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER c
    JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS o ON c.C_CUSTKEY = o.O_CUSTKEY
    JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM l ON o.O_ORDERKEY = l.L_ORDERKEY
    JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION n ON c.C_NATIONKEY = n.N_NATIONKEY
    """
    return execute_query(query)

def store_features(features_df):
    """Store engineered features in the feature store"""
    # Check if table already has data
    check_query = "SELECT COUNT(*) as count FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES;"
    result = execute_query(check_query)
    existing_count = result.iloc[0, 0]
    
    if existing_count > 0:
        print(f"Feature store already contains {existing_count} records.")
        print("Skipping feature engineering to preserve existing data.")
        return
    
    # If no data exists, proceed with insertion
    print("No existing data found. Inserting new features...")
    for _, row in features_df.iterrows():
        columns = ', '.join(row.index)
        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in row.values])
        query = f"""
        INSERT INTO FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES ({columns})
        VALUES ({values});
        """
        execute_query(query)
    print("Features stored successfully!")

def main():
    print("Loading raw data...")
    raw_data = load_raw_data()
    
    print("Performing feature engineering...")
    features = engineer_features(raw_data)
    
    print("Checking feature store...")
    store_features(features)
    
    print("Feature engineering completed!")

if __name__ == "__main__":
    main() 