import json
import tensorflow as tf
import pandas as pd
from preprocess import preprocess_inference
import joblib

# Load the model when the function starts
model_path = 'depreciation_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the scaler and encoder for inference
scaler_filename = 'scaler.pkl'
encoder_filename = 'encoder.pkl'
scaler = joblib.load(scaler_filename)
encoder = joblib.load(encoder_filename)

def handle(req: str) -> str:
    """
    Handle the incoming request from OpenFaaS.

    :param req: The request body as a JSON string
    :return: JSON string of predictions or error message
    """
    try:
        # Parse the input JSON data
        input_data = json.loads(req)

        # Preprocess the input data for inference
        X = preprocess_inference(input_data, scaler, encoder)

        # Make predictions
        predictions = model.predict(X)

        # Convert predictions to a list
        predictions_list = predictions.flatten().tolist()

        # Return the predictions as a JSON string
        return json.dumps({'predictions': predictions_list})

    except Exception as e:
        return json.dumps({'error': str(e)})


if __name__ == '__main__':
    # Local testing example
    test_input = json.dumps([{
        'screen_size': 6.1,
        'rear_camera_mp': 12,
        'front_camera_mp': 12,
        'internal_memory': 128,
        'ram': 4,
        'battery': 3000,
        'weight': 175,
        'days_used': 30,
        'device_brand': 'OnePlus',
        'os': 'Android',
        '4g': 'yes',
        '5g': 'no'
    }])
    # Call the handle function with the test input
    output = handle(test_input)
    print(output)  # Print the output for local testing