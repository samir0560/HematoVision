from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'hematovision_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/blood_model.h5'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
model = None
model_loaded = False
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_blood_model():
    """Load the trained model"""
    global model, model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            model_loaded = True
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️  Model file not found. Using dummy predictor.")
            model_loaded = False
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model_loaded = False

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_blood_cell(img_path):
    """Predict blood cell type"""
    if not model_loaded or model is None:
        # Dummy prediction for demonstration
        import random
        predicted_class = random.choice(classes)
        confidence = random.uniform(70, 95)
        return predicted_class, confidence
    
    try:
        # Preprocess image
        img_array = preprocess_image(img_path)
        if img_array is None:
            return "Error", 0
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = classes[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return "Error", 0

@app.route('/')
def home():
    """Home page route"""
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction route"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Generate unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save file
                file.save(filepath)
                
                # Make prediction
                predicted_class, confidence = predict_blood_cell(filepath)
                
                # Prepare result data
                result_data = {
                    'prediction': predicted_class,
                    'confidence': round(confidence, 2),
                    'image_path': f'/static/uploads/{unique_filename}'
                }
                
                return render_template('result.html', **result_data)
                
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                flash('Error processing image. Please try again.')
                return redirect(url_for('home'))
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(url_for('home'))
    
    # GET request - show upload form
    return render_template('home.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

def create_sample_images():
    """Create sample test images if none exist"""
    sample_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Check if samples already exist
    if len(os.listdir(sample_dir)) > 0:
        return
    
    # Create sample images
    try:
        from PIL import Image
        import random
        
        for i, class_name in enumerate(classes):
            # Create sample image for each class
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            sample_path = os.path.join(sample_dir, f'sample_{class_name.lower()}.jpg')
            img.save(sample_path)
            
        logger.info("✅ Sample images created")
    except Exception as e:
        logger.error(f"Error creating sample images: {e}")

def main():
    """Main application function"""
    print("🩸 HematoVision - Blood Cell Classification System")
    print("=" * 60)
    print("Loading model...")
    
    # Load model
    load_blood_model()
    
    # Create sample images
    create_sample_images()
    
    # Start Flask app
    print("🚀 Starting Flask application...")
    print("🌐 Access the application at: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()