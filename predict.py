import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

classes = ['Eosinophil','Lymphocyte','Monocyte','Neutrophil']

# Try to load the model, if it doesn't exist, create a dummy predictor
try:
    model = load_model("blood_model.h5")
    model_loaded = True
except:
    print("Model file not found. Using dummy predictor.")
    model_loaded = False

def predict_image(img_path):
    if not model_loaded:
        # Dummy prediction - just return a random class
        import random
        return f"Model not trained. Random prediction: {random.choice(classes)}"
    
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    return classes[np.argmax(result)]
