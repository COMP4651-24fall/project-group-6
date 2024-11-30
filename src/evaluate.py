import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model

def evaluate_model(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the model on the test data and print metrics.

    Args:
        model (tf.keras.Model): The trained model.
        X_test (np.ndarray): The test feature matrix.
        y_test (np.ndarray): The test target vector.
    """
    y_pred = model.predict(X_test)

    y_test_mean = y_test.mean()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Mean discount:", y_test_mean)
    print("Error Rate:", mae/y_test_mean)

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(range(50), y_test[0:50], label='Actual Depreciation')
    plt.plot(range(50), y_pred[0:50], label='Predicted Depreciation')
    plt.xlabel('Index')
    plt.ylabel('Depreciation')
    plt.title('Actual vs Predicted Depreciation')
    plt.legend()
    plt.show()
