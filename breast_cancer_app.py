#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('baseline_model.h5') 

# Title for your app
st.title("Breast Cancer Detection from Ultrasounds")

# Allow multiple image types for upload
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    # Convert any uploaded image to PNG for consistency
    with io.BytesIO() as buffer:
        image = Image.open(uploaded_file)
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image = Image.open(buffer)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to match your training data
    img_array = np.array(image.convert("L").resize((224, 224)))  # Convert to grayscale and resize
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)

    class_names = ['Normal', 'Benign', 'Malignant']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")

