import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_clean_data(filepath):
    """
    Loads and cleans the scoliosis data from the specified CSV path.
    """
    try:
        # Load data
        data = pd.read_csv(filepath)

        # Basic cleaning
        data.drop_duplicates(inplace=True)
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

        # Convert date columns to datetime if present
        for date_col in ["date_of_surgery", "date_of_birth"]:
            if date_col in data.columns:
                data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

def make_rf_compatible(data):
    """
    Processes the DataFrame to handle missing values and ensure compatibility with RandomForest models.
    """
    # Step 1: Handle missing values
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:  # Numeric columns
            data[col].fillna(data[col].mean(), inplace=True)
        elif data[col].dtype == "object" or data[col].dtype.name == "category":  # Categorical columns
            data[col].fillna("Unknown", inplace=True)
            data[col] = data[col].astype(str)
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        elif data[col].dtype == "bool":  # Boolean columns
            data[col].fillna(False, inplace=True)
            data[col] = data[col].astype(int)
        else:
            data[col].fillna(0, inplace=True)

    # Ensure all non-numeric columns are converted to numeric
    for col in data.select_dtypes(include=["object", "category"]).columns:
        data[col] = data[col].astype(str)
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Check for any remaining NaN values
    if data.isna().any().any():
        print("Warning: NaN values remain after processing.")
    else:
        print("Data cleaned and compatible with RandomForest.")

    return data

def prepare_train_test(data, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def run_random_forest_regression(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    """
    Fits a Random Forest Regressor and evaluates it on the test set.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Regression Results:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

def main():
    """
    Main entry point for training a Random Forest model on the scoliosis dataset.
    """
    if len(sys.argv) < 2:
        print("Usage: python3 scoliosis_rf.py <csv_filepath> [<target_column>]")
        sys.exit(1)

    # Parse command-line arguments
    filepath = sys.argv[1]
    target_column = sys.argv[2] if len(sys.argv) > 2 else "tothlos"

    print(f"Loading data from: {filepath}")
    data = load_and_clean_data(filepath)

    # Ensure the target column exists
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        sys.exit(1)

    # Preprocess data
    print("Preprocessing data...")
    data = make_rf_compatible(data)

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = prepare_train_test(data, target_column)

    # Train and evaluate the Random Forest model
    print("Training Random Forest model...")
    run_random_forest_regression(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()