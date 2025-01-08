# data_cleaning.py

import pandas as pd

def drop_rows_with_excessive_missing(dataframe, max_missing_columns=4):
    """
    Remove rows with more than `max_missing_columns` missing values.
    """
    filtered_df = dataframe[dataframe.isnull().sum(axis=1) <= max_missing_columns]
    return filtered_df

def impute_missing_values(dataframe):
    """
    Impute missing values: 
        - Numerical columns: Replace missing values with column mean.
        - Categorical columns: Replace missing values with column mode.
    """
    # Separate numeric and categorical columns
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = dataframe.select_dtypes(include=['object']).columns
    
    # Replace missing values ​​in numeric columns with the mean
    for col in numerical_cols:
        mean_value = dataframe[col].mean()
        dataframe[col].fillna(mean_value, inplace=True)
    
    # Replace missing values ​​in categorical columns with mode
    for col in categorical_cols:
        mode_value = dataframe[col].mode()[0]
        dataframe[col].fillna(mode_value, inplace=True)
    
    return dataframe
