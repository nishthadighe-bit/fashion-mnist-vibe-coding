import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image of a clothing item!")

file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if file:
    img = Image.open(file).convert('L').resize((28, 28))
    st.image(img)
    st.write("Processing... (This is your Vibe project in action!)")
