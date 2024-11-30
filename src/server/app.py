import tensorflow as tf
from preprocess import preprocess_inference
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model when the function starts
model_path = 'depreciation_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the scaler and encoder for inference
scaler_filename = 'scaler.pkl'
encoder_filename = 'encoder.pkl'
scaler = joblib.load(scaler_filename)
encoder = joblib.load(encoder_filename)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json[0]
    X = preprocess_inference(data, scaler, encoder)
    if not X:
        return jsonify({'error': 'No features provided'}), 400

    prediction = model.predict(X)  # 将特征转换为模型所需的格式
    return jsonify({'prediction': prediction.flatten().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 监听5000端口
