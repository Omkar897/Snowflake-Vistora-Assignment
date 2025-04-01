# Snowflake Feature Engineering and Model Training

This project demonstrates a complete workflow for:

1. Connecting to Snowflake
2. Extracting data
3. Performing feature engineering
4. Storing features in Snowflake
5. Training a machine learning model

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Configure your Snowflake credentials:
   - Copy the `.env` file and fill in your Snowflake credentials
   - Make sure to keep your credentials secure and never commit them to version control

## Usage

1. Modify the `sample_query` in `main.py` to point to your actual data table
2. Adjust the feature engineering logic in `snowflake_utils.py` based on your data
3. Update the target variable name in `main.py` to match your data
4. Run the main script:

```bash
python main.py
```

## Project Structure

- `snowflake_utils.py`: Contains utility functions for Snowflake operations and feature engineering
- `main.py`: Main script demonstrating the complete workflow
- `.env`: Configuration file for Snowflake credentials
- `requirements.txt`: Project dependencies

## Customization

You can customize the feature engineering process by modifying the `engineer_features` function in `snowflake_utils.py`. The current implementation includes:

- Basic feature scaling
- Example derived features (squared and log transformations)
- You can add your own feature engineering steps based on your data
