import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_dataset_info(dataframe):
    """
    Function to display basic information and statistical description of a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Display dataset structure, types, and non-null counts
    print("Dataset Information:")
    print(dataframe.info())
    
    # Display statistical summary for all columns, including categorical and numerical features
    print("\nStatistical Description:")
    print(dataframe.describe(include='all'))


def display_missing_values(dataframe):
    """
    Function to display the count of missing values for each column in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Calculate the number of missing values for each column
    missing_values = dataframe.isnull().sum()
    
    # Print the missing values count for each column
    print("\nMissing Values by Column:")
    print(missing_values)

def display_missing_percentage(dataframe):
    """
    Function to calculate and display the percentage of missing values for each column in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Calculate the number of missing values for each column
    missing_values = dataframe.isnull().sum()
    
    # Calculate the percentage of missing values for each column
    missing_percentage = (missing_values / len(dataframe)) * 100
    
    # Print the percentage of missing values for each column
    print("\nPercentage of Missing Values by Column:")
    print(missing_percentage)

def display_numerical_analysis(dataframe):
    """
    Function to display a statistical summary of numerical features in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Select numerical columns (float64)
    numerical_cols = dataframe.select_dtypes(include=['float64']).columns
    
    # Display a statistical summary for numerical features
    print("\nStatistical Analysis of Numerical Features:")
    print(dataframe[numerical_cols].describe())

def plot_numerical_boxplots(dataframe):
    """
    Function to create and display boxplots for numerical features in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Select numerical columns (float64)
    numerical_cols = dataframe.select_dtypes(include=['float64']).columns
    
    # Loop through each numerical column to create a boxplot
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=dataframe[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.show()

def plot_numerical_distributions(dataframe):
    """
    Function to create and display histograms with KDE for numerical features in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Select numerical columns (float64)
    numerical_cols = dataframe.select_dtypes(include=['float64']).columns
    
    # Loop through each numerical column to create a histogram with KDE
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(dataframe[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def display_categorical_analysis(dataframe):
    """
    Function to display the frequency analysis of categorical features in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Select categorical columns (object type)
    categorical_cols = dataframe.select_dtypes(include=['object']).columns
    
    # Loop through each categorical column to display unique values and their frequencies
    print("\nFrequency Analysis of Categorical Features:")
    for col in categorical_cols:
        print(f"\n{col} - Unique Values and Frequencies:")
        print(dataframe[col].value_counts())

def plot_categorical_distributions(dataframe):
    """
    Function to create and display count plots for categorical features in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    # Select categorical columns (object type)
    categorical_cols = dataframe.select_dtypes(include=['object']).columns
    
    # Loop through each categorical column to create a count plot
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=dataframe[col], order=dataframe[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

def plot_correlation_matrix(dataframe):
    """
    Function to calculate and display the correlation matrix for numerical features in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.

    Returns:
        None
    """
    #import matplotlib.pyplot as plt
    #import seaborn as sns

    # Select numerical columns (float64)
    numerical_cols = dataframe.select_dtypes(include=['float64']).columns
    
    # Calculate the correlation matrix
    correlation_matrix = dataframe[numerical_cols].corr()
    
    # Display the correlation matrix as a heatmap
    print("\nCorrelation between Numerical Features:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Map")
    plt.show()

def analyze_target_variable(dataframe, target_col):
    """
    Function to display the distribution of the target variable and plot a count plot.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
        target_col (str): The name of the target variable column.

    Returns:
        None
    """
    #import matplotlib.pyplot as plt
    #import seaborn as sns

    # Display the distribution of the target variable
    print("\nTarget Variable Distribution:")
    print(dataframe[target_col].value_counts())
    
    # Plot the count plot for the target variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x=dataframe[target_col])
    plt.title("Distribution of the Target Variable")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.show()
