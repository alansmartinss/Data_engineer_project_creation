import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




def replace_yes_no(dataframe, columns):
    """
    Function to replace "Yes" with 1 and "No" with 0 in specified columns of a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
        columns (list): List of column names to apply the transformation.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns transformed.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    transformed_df = dataframe.copy()

    # Iterate over the specified columns and replace "Yes" and "No"
    for col in columns:
        if col in transformed_df.columns:
            transformed_df[col] = transformed_df[col].replace({"Yes": 1, "No": 0})
        else:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    return transformed_df

def encode_categorical_to_numpy(dataframe, categorical_columns, other_columns):
    """
    Function to encode categorical features into numeric arrays for each row
    and combine them with other columns.

    Parameters:
        dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
        categorical_columns (list): List of categorical columns to be encoded.
        other_columns (list): List of other (non-categorical) columns.

    Returns:
        np.ndarray: Combined NumPy array with encoded categorical and numerical features.
    """
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform the categorical columns
    encoded_data = encoder.fit_transform(dataframe[categorical_columns])

    # Select other (numerical) columns
    numerical_data = dataframe[other_columns].values

    # Combine the encoded categorical data with numerical data
    combined_data = np.hstack([encoded_data, numerical_data])

    return combined_data, encoder

def evaluate_regression_model(model, X_test, y_test, y_pred):
    """
    Evaluate the regression model using statistical metrics and visualizations.

    Parameters:
        model (LinearRegression): Trained regression model.
        X_test (array): Features of the test set.
        y_test (array): True target values of the test set.
        y_pred (array): Predicted target values from the model.

    Returns:
        None
    """
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    # R^2 Score
    r2 = r2_score(y_test, y_pred)
    # Explained Variance
    explained_variance = 1 - (np.var(y_test - y_pred) / np.var(y_test))
    
    # Display metrics
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Explained Variance: {explained_variance:.2f}")

    # Residuals (Errors)
    residuals = y_test - y_pred

    # Plot: Residuals Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.legend()
    plt.show()

    # Plot: Predicted vs True
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.title("Predicted vs True Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()

    # Plot: Residuals vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color='blue', edgecolors='k')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()

def evaluate_decision_tree_model(model, X_test, y_test, y_pred, feature_names=None):
    """
    Evaluate the decision tree regression model using statistical metrics and visualizations.

    Parameters:
        model (DecisionTreeRegressor): Trained decision tree regression model.
        X_test (array or DataFrame): Features of the test set.
        y_test (array): True target values of the test set.
        y_pred (array): Predicted target values from the model.
        feature_names (list, optional): List of feature names. Required if X_test is a NumPy array.

    Returns:
        None
    """
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    # R^2 Score
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    print("Decision Tree Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Residuals (Errors)
    residuals = y_test - y_pred

    # Plot: Residuals Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='green')
    plt.title("Residuals Distribution (Decision Tree)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.legend()
    plt.show()

    # Plot: Predicted vs True Values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='green', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.title("Predicted vs True Values (Decision Tree)")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()

    # Plot: Residuals vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color='green', edgecolors='k')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title("Residuals vs Predicted Values (Decision Tree)")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()