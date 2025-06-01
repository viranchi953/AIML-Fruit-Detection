from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from keras.models import load_model
import base64
from io import BytesIO
import uuid

# Fix for DepthwiseConv2D compatibility issue
from tensorflow.keras.utils import get_custom_objects

def custom_depthwise_conv2d(*args, **kwargs):
    # Remove the 'groups' argument if present
    if 'groups' in kwargs:
        del kwargs['groups']
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

# Register the custom object
get_custom_objects()['DepthwiseConv2D'] = custom_depthwise_conv2d

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Global variables for model and labels
model = None
class_names = []

def load_ai_model():
    """Load the Keras model and labels with compatibility fixes"""
    global model, class_names
    try:
        print("Loading model with compatibility fixes...")
        
        # Load labels first
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Try different loading approaches for better compatibility
        try:
            # Attempt 1: Load with compile=False and custom objects
            custom_objects = {
                'DepthwiseConv2D': custom_depthwise_conv2d
            }
            model = tf.keras.models.load_model("keras_Model.h5", compile=False, custom_objects=custom_objects)
            
        except Exception as e1:
            print(f"First loading attempt failed: {e1}")
            try:
                # Attempt 2: Standard loading with TensorFlow
                model = tf.keras.models.load_model("keras_Model.h5", compile=False)
                
            except Exception as e2:
                print(f"Second loading attempt failed: {e2}")
                try:
                    # Attempt 3: Using Keras load_model
                    model = load_model("keras_Model.h5", compile=False)
                    
                except Exception as e3:
                    print(f"All loading attempts failed. Last error: {e3}")
                    raise Exception("Unable to load model with current TensorFlow/Keras version")
        
        print("Model and labels loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {[name.split(' ', 1)[-1] if ' ' in name else name for name in class_names]}")
        return True
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure 'keras_Model.h5' and 'labels.txt' are in the same directory as app.py")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"TensorFlow version: {tf.__version__}")
        print("\nTroubleshooting suggestions:")
        print("1. Try updating TensorFlow: pip install --upgrade tensorflow")
        print("2. Or try downgrading: pip install tensorflow==2.12.0")
        print("3. Recreate the model with your current TensorFlow version")
        return False

def predict_fruit(image_path):
    """Predict fruit type from image"""
    try:
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        
        # Resize the image to be at least 224x224 and then crop from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Turn the image into a numpy array
        image_array = np.asarray(image)
        
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Load the image into the array
        data[0] = normalized_image_array
        
        # Make prediction
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        
        # Extract class name (remove the number prefix like "0 Apple" -> "Apple")
        class_name = class_names[index].split(' ', 1)[1] if ' ' in class_names[index] else class_names[index]
        confidence_score = float(prediction[0][index])
        
        return {
            'success': True,
            'fruit': class_name,
            'confidence': confidence_score,
            'all_predictions': {
                class_names[i].split(' ', 1)[1] if ' ' in class_names[i] else class_names[i]: float(prediction[0][i]) 
                for i in range(len(class_names))
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/detect')
def detect_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_fruit(filepath)
            
            # Convert image to base64 for display
            if result['success']:
                with open(filepath, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                result['image'] = f"data:image/jpeg;base64,{img_base64}"
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Load model on startup
    if load_ai_model():
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check if keras_Model.h5 and labels.txt exist.")