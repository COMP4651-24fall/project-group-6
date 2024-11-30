import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from model import create_model
from evaluate import evaluate_model
import tensorflow as tf

def train_model(data_file: str, model_save_path: str) -> None:
    """
    Train the model and save it as an .h5 file.

    :param data_file: Path to the data file for training
    :param model_save_path: Path to save the model
    """
    # Preprocess the data
    X, y = preprocess_data(data_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Create the model
    model = create_model(X_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=500, batch_size=2000, verbose=1)

    # Make predictions and evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    train_model('data/used_device_data.csv', 'depreciation_model.h5')