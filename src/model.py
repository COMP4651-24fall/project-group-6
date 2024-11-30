import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(input_shape: int) -> Sequential:
    """
    Create a Keras Sequential model.

    :param input_shape: The number of input features
    :return: Compiled Keras model
    """
    model = Sequential()
    model.add(Dense(7, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model