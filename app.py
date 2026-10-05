from pathlib import Path
from threading import Lock
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 11 * 1024 * 1024
MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'best_efficientnet.keras'
model = None
model_lock = Lock()

@app.get('/')
def home():
    return render_template('index.html')

@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Please upload an image smaller than 10 MB.'), 413

@app.post('/predict')
def predict():
    global model
    uploaded = request.files.get('image')
    if uploaded is None or not uploaded.filename:
        return jsonify(error='Please select an image.'), 400
    try:
        image = Image.open(uploaded.stream)
        if image.format not in ('JPEG', 'PNG'):
            return jsonify(error='Only JPG and PNG images are supported.'), 400
        image = image.convert('RGB')
        # Same resize and preprocessing as the original Streamlit app.
        pixels = np.array(image.resize((224, 224)), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error='Could not read this image. Choose a valid JPG or PNG.'), 400
    try:
        if not MODEL_PATH.is_file():
            return jsonify(error='Model missing: place best_efficientnet.keras in the model folder next to app.py.'), 503
        import tensorflow as tf
        from tensorflow.keras.applications.efficientnet import preprocess_input
        pixels = np.expand_dims(preprocess_input(pixels), axis=0)
        with model_lock:
            if model is None:
                model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            output = np.asarray(model.predict(pixels, verbose=0))
        if output.size != 1:
            raise ValueError('Expected one binary output score.')
        score = float(output.reshape(-1)[0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score.')
        # Original class mapping: fake=0, real=1; threshold=0.5.
        is_real = score >= 0.5
        return jsonify(label='REAL FACE' if is_real else 'AI-GENERATED / FAKE FACE',
                       real_score=score, fake_score=1-score,
                       class_score=score if is_real else 1-score, is_real=is_real)
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify(error='Analysis failed. Check the terminal for model or TensorFlow errors.'), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
