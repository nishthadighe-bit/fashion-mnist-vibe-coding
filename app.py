import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Fashion MNIST Classifier")

# 1. Define the 10 categories the model knows
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if file:
    img = Image.open(file).convert('L').resize((28, 28))
    st.image(img, caption="Resized Image (28x28)")
    
    # 2. Preprocess: Normalize and Reshape for the model
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # 3. Load Model & Predict
    # Note: Ensure your model file (e.g., model.h5) is in the same GitHub folder!
    try:
        model = tf.keras.models.load_model('model.h5') # Change name to match your file
        prediction = model.predict(img_array)
        result = class_names[np.argmax(prediction)]
        
        # 4. Display the Final Result!
        st.success(f"Prediction: This is a **{result}**!")
    except Exception as e:
        st.error("Model file not found. Make sure 'model.h5' is in your GitHub!")
