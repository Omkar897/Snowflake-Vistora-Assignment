import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.snowflake_utils import execute_query

def test_connection():
    """Test Snowflake connection and basic queries"""
    try:
        # Test basic query
        print("Testing basic query...")
        result = execute_query("SELECT CURRENT_VERSION()")
        print(f"Connected to Snowflake! Version: {result.iloc[0, 0]}")
        
        # Test database access
        print("\nTesting database access...")
        result = execute_query("SHOW DATABASES")
        print("Available databases:")
        for db in result['name']:
            print(f"- {db}")
        
        # Test feature store access
        print("\nTesting feature store access...")
        result = execute_query("SELECT COUNT(*) FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES")
        print(f"Number of records in feature store: {result.iloc[0, 0]}")
        
        print("\nAll connection tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during connection test: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 