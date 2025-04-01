import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.snowflake_utils import execute_query
import pandas as pd

def explore_feature_store():
    """Explore the feature store structure and data"""
    # Get table information
    print("Feature Store Table Structure:")
    print("-----------------------------") 
    result = execute_query("""
    DESCRIBE TABLE FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES;
    """)
    print(result)
    
    # Get sample data
    print("\nSample Data:")
    print("------------")
    result = execute_query("""
    SELECT * FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES
    LIMIT 5;
    """)
    print(result)
    
    # Get basic statistics
    print("\nBasic Statistics:")
    print("----------------")
    result = execute_query("""
    SELECT 
        COUNT(*) as total_customers,
        COUNT(DISTINCT MARKET_SEGMENT) as unique_markets,
        COUNT(DISTINCT NATION) as unique_nations,
        AVG(TOTAL_ORDERS) as avg_orders,
        AVG(AVG_ORDER_VALUE) as avg_order_value,
        AVG(CUSTOMER_LIFETIME_DAYS) as avg_lifetime_days
    FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES;
    """)
    print(result)
    
    # Get segment distribution
    print("\nMarket Segment Distribution:")
    print("---------------------------")
    result = execute_query("""
    SELECT 
        MARKET_SEGMENT,
        COUNT(*) as customer_count,
        AVG(AVG_DAILY_SPEND) as avg_daily_spend
    FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES
    GROUP BY MARKET_SEGMENT
    ORDER BY customer_count DESC;
    """)
    print(result)

if __name__ == "__main__":
    explore_feature_store() 