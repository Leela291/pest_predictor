from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('crop_pest_model.h5')

# These should match the folder names in your dataset
class_names = ['healthy', 'aphid', 'mite', 'thrips', 'whitefly', 'bollworm', 'leafminer']

# Pesticide info per pest
pesticide_info = {
    'aphid': {
        'pesticide': 'Imidacloprid 17.8% SL',
        'dosage': '1 ml per liter of water',
        'interval': 'Every 10–14 days'
    },
    'mite': {
        'pesticide': 'Abamectin 1.8% EC',
        'dosage': '0.5 ml per liter of water',
        'interval': 'Every 7–10 days'
    },
    'thrips': {
        'pesticide': 'Spinosad 45% SC',
        'dosage': '1 ml per liter of water',
        'interval': 'Every 7 days'
    },
    'whitefly': {
        'pesticide': 'Acetamiprid 20% SP',
        'dosage': '0.75 g per liter of water',
        'interval': 'Every 10–15 days'
    },
    'bollworm': {
        'pesticide': 'Chlorantraniliprole 18.5% SC',
        'dosage': '0.4 ml per liter of water',
        'interval': 'Apply at egg-laying stage, then as needed'
    },
    'leafminer': {
        'pesticide': 'Neem Oil 3%',
        'dosage': '2 ml per liter of water',
        'interval': 'Every 5–7 days'
    }
}

# Image preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert("RGB")
            input_img = preprocess_image(image)
            pred = model.predict(input_img)
            class_index = np.argmax(pred[0])
            class_label = class_names[class_index]

            if class_label == 'healthy':
                result = "✅ The crop is healthy! No pesticide required."
            elif class_label in pesticide_info:
                info = pesticide_info[class_label]
                result = (f"⚠️ The crop is affected by <b>{class_label}</b>.<br>"
                          f"💊 <b>Recommended pesticide:</b> {info['pesticide']}<br>"
                          f"🧪 <b>Dosage:</b> {info['dosage']}<br>"
                          f"⏱️ <b>Spray interval:</b> {info['interval']}")
            else:
                result = f"⚠️ Pest detected: <b>{class_label}</b>. No pesticide info available."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
