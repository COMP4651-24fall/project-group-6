import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple

# Global variables for scaler and encoder file names
scaler_filename = 'scaler.pkl'
encoder_filename = 'encoder.pkl'

def preprocess_data(data_path: str) -> Tuple[np.ndarray, pd.Series]:
    """Preprocess the input data for training.

    Args:
        data_path (str): The path to the CSV file containing the data.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the target vector (y).
    """
    rdata = pd.read_csv(data_path)
    rdata.dropna(inplace=True)
    rdata['Depreciation rate'] = (rdata['normalized_new_price'] - rdata['normalized_used_price']) / rdata['normalized_new_price']

    # Drop unnecessary columns
    data = rdata.drop(['release_year', 'normalized_used_price', 'normalized_new_price'], axis=1)

    X = data.drop('Depreciation rate', axis=1).copy()
    y = data['Depreciation rate']

    # Normalize numerical features
    numerical_features = ['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 
                          'ram', 'battery', 'weight', 'days_used']
    scaler = MinMaxScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Encode categorical features
    categorical_features = ['device_brand', 'os', '4g', '5g']
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_features]).toarray()

    # Combine encoded categorical features with normalized numerical features
    X = np.hstack((X_encoded, X[numerical_features].values))

    # Save the scaler and encoder for later use
    joblib.dump(scaler, scaler_filename)
    joblib.dump(encoder, encoder_filename)

    return X, y

def preprocess_inference(input_data: dict, 
                         scaler: MinMaxScaler, 
                         encoder: OneHotEncoder) -> np.ndarray:
    """Preprocess the input data for inference.

    Args:
        input_data (dict): A dictionary containing the input data.
        scaler (MinMaxScaler): The scaler used for normalizing numerical features.
        encoder (OneHotEncoder): The encoder used for encoding categorical features.

    Returns:
        np.ndarray: The preprocessed feature matrix (X) ready for prediction.
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Convert numerical columns to appropriate types
    numerical_features = ['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 
                          'ram', 'battery', 'weight', 'days_used']
    for feature in numerical_features:
        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce')
    
    # Fill missing values in numerical features with the mean
    input_df[numerical_features] = input_df[numerical_features].fillna(input_df[numerical_features].mean())

    # Ensure categorical features are strings and fill missing values
    categorical_features = ['device_brand', 'os', '4g', '5g']
    for feature in categorical_features:
        input_df[feature] = input_df[feature].astype(str)  # Convert to string
        input_df[feature] = input_df[feature].fillna('unknown')  # Fill missing values with a placeholder
    # Normalize numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    # Encode categorical features
    encoded_features = encoder.transform(input_df[categorical_features]).toarray()
    # Combine encoded categorical features with normalized numerical features
    X = np.hstack((encoded_features, input_df[numerical_features].values))
    return X