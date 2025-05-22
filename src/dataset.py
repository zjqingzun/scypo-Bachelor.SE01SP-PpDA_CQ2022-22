# Import necessary libraries
import pandas as pd

# 1. Load the datasets
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/test.csv')

# 2. Print information about train.csv
print("Number of rows in train.csv:", len(train_df))
print("Column names in train.csv:", train_df.columns.tolist())

# 3. Display the first 5 rows of the training dataset
print("\nFirst 5 rows of train.csv:")
print(train_df.head())