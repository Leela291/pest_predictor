from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB file size limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = tf.keras.models.load_model('crop_pest_model.h5')

# Class names and pesticide info (keep your existing data)
class_names = ['healthy', 'aphid', 'mite', 'thrips', 'whitefly', 'bollworm', 'leafminer']
pesticide_info = {
    # ... (keep your existing pesticide_info dictionary) ...
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', result="⚠️ No file selected")
            
        file = request.files['image']
        
        if file.filename == '':
            return render_template('index.html', result="⚠️ No file selected")
            
        if not allowed_file(file.filename):
            return render_template('index.html', 
                                result="⚠️ Invalid file type. Please upload JPG, JPEG, or PNG")

        try:
            filename = secure_filename(file.filename)
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
                
        except Exception as e:
            result = f"⚠️ Error processing image: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)