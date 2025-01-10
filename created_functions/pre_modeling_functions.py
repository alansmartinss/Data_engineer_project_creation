import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def plot_target_percentage(dataframe, target_col):
    """
    Function to plot the percentage distribution of the target variable.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
        target_col (str): The name of the target variable column.

    Returns:
        None
    """
    # Calculate percentage distribution of the target variable
    target_percentage = (dataframe[target_col].value_counts(normalize=True) * 100).round(2)

    # Plot the percentage distribution
    plt.figure(figsize=(6, 4))
    target_percentage.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Percentage Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_numeric_histogram(dataframe, column, bins=30):
    """
    Function to plot a histogram for a numerical column in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
        column (str): The name of the numerical column to plot.
        bins (int): The number of bins for the histogram (default is 30).

    Returns:
        None
    """
    # Check if the column exists in the DataFrame
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Check if the column is numerical
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ValueError(f"Column '{column}' is not numerical.")

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(dataframe[column], bins=bins, kde=True, color='skyblue')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def compare_histograms(dataset1, dataset2, column, bins=30, labels=('Dataset 1', 'Dataset 2')):
    """
    Function to overlay histograms of the same column from two datasets for comparison.

    Parameters:
        dataset1 (pd.DataFrame): The first dataset as a pandas DataFrame.
        dataset2 (pd.DataFrame): The second dataset as a pandas DataFrame.
        column (str): The name of the numerical column to compare.
        bins (int): The number of bins for the histograms (default is 30).
        labels (tuple): Labels for the datasets in the legend (default is ('Dataset 1', 'Dataset 2')).

    Returns:
        None
    """
    # Check if the column exists in both datasets
    if column not in dataset1.columns or column not in dataset2.columns:
        raise ValueError(f"Column '{column}' not found in one or both datasets.")

    # Check if the column is numerical in both datasets
    if not (pd.api.types.is_numeric_dtype(dataset1[column]) and pd.api.types.is_numeric_dtype(dataset2[column])):
        raise ValueError(f"Column '{column}' is not numerical in one or both datasets.")

    # Plot the histograms
    plt.figure(figsize=(8, 6))
    sns.histplot(dataset1[column], bins=bins, kde=True, color='blue', label=labels[0], alpha=0.6)
    sns.histplot(dataset2[column], bins=bins, kde=True, color='orange', label=labels[1], alpha=0.6)
    plt.title(f"Comparison of {column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def preprocess_dataset(dataframe):
    """
    Function to preprocess a dataset by normalizing numerical columns and encoding categorical columns.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        pd.DataFrame: A transformed DataFrame with normalized numerical and encoded categorical features.
    """
    # Separate features into numerical and categorical
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = dataframe.select_dtypes(include=['object']).columns

    # Define transformations for numerical and categorical columns
    transformers = [
        ('num', MinMaxScaler(), numerical_cols),           # Normalize numerical columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical columns
    ]

    # Create ColumnTransformer
    column_transformer = ColumnTransformer(transformers)

    # Apply transformations to the dataset
    transformed_data = column_transformer.fit_transform(dataframe)

    # Create a DataFrame with transformed data
    transformed_columns = (
        list(numerical_cols) + 
        list(column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    )
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns, index=dataframe.index)

    return transformed_df