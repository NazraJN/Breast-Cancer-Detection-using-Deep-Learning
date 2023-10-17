# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:10:02 2023

@author: hp
"""

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:/Users/hp/Desktop/Moringa_practice/Deployment/baseline_model.h5')

def classify_image(uploaded_image):
    img = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=[0, -1])
    img_data = img_array / 255.0  # Normalize

    prediction = model.predict(img_data)
    return prediction

st.title("Deep-Learning-Based Breast Cancer Prediction System")
uploaded_image = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image:
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    with st.spinner('Classifying...'):
        # For demonstration purposes, let's simulate some processing steps
        import time
        for i in range(4):
            # simulate a portion of the processing
            time.sleep(0.5)
        
    prediction = classify_image(uploaded_image)
    class_names = ['Normal', 'Benign', 'Malignant']
    predicted_class = class_names[np.argmax(prediction)]
    
    st.write(f"Prediction: {predicted_class}")
    
st.markdown("**DISCLAIMER**")
st.markdown("""
This application is designed for educational and informational purposes only. The predictions provided by this tool should NOT be used as a substitute for professional medical advice or diagnosis. Always consult your physician or another qualified healthcare provider with any questions you may have regarding a medical condition. Do not disregard professional medical advice or delay in seeking it because of something you have read or interpreted from this application's results.

Relying on this application for medical decision-making is strictly at your own risk. The developers, contributors, and stakeholders associated with this application are not responsible for any claim, loss, or damage arising from the use of this tool.
""")

