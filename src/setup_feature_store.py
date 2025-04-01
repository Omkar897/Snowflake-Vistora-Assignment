import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.snowflake_utils import execute_query

def create_feature_store():
    """Create the feature store database and tables if they don't exist"""
    # Check if database exists
    check_db_query = "SHOW DATABASES LIKE 'FEATURE_STORE_DB';"
    db_exists = len(execute_query(check_db_query)) > 0
    
    if db_exists:
        print("Feature store database already exists.")
        print("Checking tables...")
    else:
        print("Creating Feature Store database...")
        execute_query("""
        CREATE DATABASE IF NOT EXISTS FEATURE_STORE_DB;
        """)
        print("Database created successfully!")
    
    # Check if schema exists
    check_schema_query = """
    SELECT COUNT(*) as count 
    FROM FEATURE_STORE_DB.INFORMATION_SCHEMA.SCHEMATA 
    WHERE SCHEMA_NAME = 'PUBLIC';
    """
    schema_exists = execute_query(check_schema_query).iloc[0, 0] > 0
    
    if not schema_exists:
        print("Creating schema...")
        execute_query("""
        CREATE SCHEMA IF NOT EXISTS FEATURE_STORE_DB.PUBLIC;
        """)
        print("Schema created successfully!")
    
    # Check if table exists
    check_table_query = """
    SELECT COUNT(*) as count 
    FROM FEATURE_STORE_DB.INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'PUBLIC' 
    AND TABLE_NAME = 'CUSTOMER_FEATURES';
    """
    table_exists = execute_query(check_table_query).iloc[0, 0] > 0
    
    if table_exists:
        print("Customer features table already exists.")
        print("Setup completed - no changes needed.")
        return
    
    print("Creating customer features table...")
    execute_query("""
    CREATE TABLE IF NOT EXISTS FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES (
        CUSTOMER_ID VARCHAR,
        TOTAL_ORDERS INTEGER,
        AVG_ORDER_VALUE FLOAT,
        CUSTOMER_LIFETIME_DAYS INTEGER,
        TOTAL_ITEMS_ORDERED INTEGER,
        AVG_QUANTITY_PER_ITEM FLOAT,
        AVG_DISCOUNT FLOAT,
        AVG_DAILY_SPEND FLOAT,
        ORDERS_PER_MONTH FLOAT,
        ITEMS_PER_ORDER FLOAT,
        MARKET_SEGMENT VARCHAR,
        NATION VARCHAR,
        SPENDING_SEGMENT VARCHAR,
        FREQUENCY_SEGMENT VARCHAR,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """)
    print("Table created successfully!")
    print("Feature store setup completed!")

def load_sample_data():
    """Load sample data into the feature store"""
    # First check if table has data
    check_data_query = "SELECT COUNT(*) as count FROM FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES;"
    has_data = execute_query(check_data_query).iloc[0, 0] > 0
    
    if has_data:
        print("Feature store already contains data.")
        print("Skipping sample data load to preserve existing data.")
        return
        
    print("Loading sample data...")
    
    # Sample data for testing
    sample_data = [
        {
            'CUSTOMER_ID': '75872',
            'TOTAL_ORDERS': 45,
            'AVG_ORDER_VALUE': 1250.50,
            'CUSTOMER_LIFETIME_DAYS': 365,
            'TOTAL_ITEMS_ORDERED': 180,
            'AVG_QUANTITY_PER_ITEM': 4,
            'AVG_DISCOUNT': 0.15,
            'AVG_DAILY_SPEND': 154.25,
            'ORDERS_PER_MONTH': 3.75,
            'ITEMS_PER_ORDER': 4,
            'MARKET_SEGMENT': 'MACHINERY',
            'NATION': 'ALGERIA',
            'SPENDING_SEGMENT': 'HIGH',
            'FREQUENCY_SEGMENT': 'REGULAR'
        },
        {
            'CUSTOMER_ID': '4796',
            'TOTAL_ORDERS': 32,
            'AVG_ORDER_VALUE': 980.75,
            'CUSTOMER_LIFETIME_DAYS': 240,
            'TOTAL_ITEMS_ORDERED': 128,
            'AVG_QUANTITY_PER_ITEM': 4,
            'AVG_DISCOUNT': 0.10,
            'AVG_DAILY_SPEND': 130.75,
            'ORDERS_PER_MONTH': 4,
            'ITEMS_PER_ORDER': 4,
            'MARKET_SEGMENT': 'BUILDING',
            'NATION': 'UNITED KINGDOM',
            'SPENDING_SEGMENT': 'MEDIUM',
            'FREQUENCY_SEGMENT': 'REGULAR'
        }
    ]
    
    # Insert sample data
    for customer in sample_data:
        columns = ', '.join(customer.keys())
        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in customer.values()])
        query = f"""
        INSERT INTO FEATURE_STORE_DB.PUBLIC.CUSTOMER_FEATURES ({columns})
        VALUES ({values});
        """
        execute_query(query)
    
    print("Sample data loaded successfully!")

def main():
    print("Setting up feature store...")
    create_feature_store()
    load_sample_data()

if __name__ == "__main__":
    main() 