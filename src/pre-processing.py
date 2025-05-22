# Import necessary libraries
import pandas as pd
import numpy as np

# Load the training dataset
train_df = pd.read_csv('./data/train.csv')

# 1. Select relevant columns for the model
selected_columns = ['Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Max Power', 'Max Torque', 'Seating Capacity', 'Fuel Tank Capacity']
X = train_df[selected_columns]
y = train_df['Price']  # Assuming 'Price' is the column name for Y

# 2. Preprocess the data
# Handle categorical data (Fuel Type, Transmission)
fuel_type_mapping = {fuel: idx + 1 for idx, fuel in enumerate(X['Fuel Type'].unique())}
transmission_mapping = {'manual': 1, 'auto': 2}

# Apply mappings
X = X.copy()  # Avoid SettingWithCopyWarning
X['Fuel Type'] = X['Fuel Type'].map(fuel_type_mapping)
X['Transmission'] = X['Transmission'].map(transmission_mapping)

# Handle numeric data with units (e.g., Max Power, Max Torque)
def extract_numeric(value):
    try:
        return float(value.split()[0])  # Extract number before unit
    except:
        return np.nan  # Handle missing or malformed data

X['Max Power'] = X['Max Power'].apply(extract_numeric)
X['Max Torque'] = X['Max Torque'].apply(extract_numeric)

# Handle missing values by filling with median for numeric columns
numeric_columns = ['Year', 'Kilometer', 'Max Power', 'Max Torque', 'Seating Capacity', 'Fuel Tank Capacity']
for col in numeric_columns:
    X[col] = X[col].fillna(X[col].median())

# Handle missing values for categorical columns
# Check if mode exists before filling
if not X['Fuel Type'].isna().all() and not X['Fuel Type'].mode().empty:
    X['Fuel Type'] = X['Fuel Type'].fillna(X['Fuel Type'].mode()[0])
else:
    X['Fuel Type'] = X['Fuel Type'].fillna(1)  # Default value (e.g., first category)

if not X['Transmission'].isna().all() and not X['Transmission'].mode().empty:
    X['Transmission'] = X['Transmission'].fillna(X['Transmission'].mode()[0])
else:
    X['Transmission'] = X['Transmission'].fillna(1)  # Default value (e.g., manual)

# Print the preprocessed dataset's first 5 rows
print("Preprocessed dataset (first 5 rows):")
print(X.head())

# Save preprocessed data for model training
X.to_csv('./data/preprocessed_train.csv', index=False)
y.to_csv('./data/preprocessed_train_y.csv', index=False)