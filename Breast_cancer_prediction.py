#!/usr/bin/env python
# coding: utf-8

# In[12]:


from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
from PIL import Image

# Load the trained model
saved_model = load_model('./baseline_model.h5')

# Image path
image_path = ('benign.jpeg')  # ext can be any image extension

# Convert any image to PNG for consistent processing
with open(image_path, "rb") as f:
    with Image.open(f) as img:
        img_format = img.format
        if img_format != 'PNG':
            png_path = os.path.splitext(image_path)[0] + '.png'
            img.save(png_path, 'PNG')
            image_path = png_path

# Load and preprocess the image in grayscale
img = image.load_img(image_path, color_mode='grayscale', target_size=(224, 224))
image_array = image.img_to_array(img)
image_array = np.expand_dims(image_array, axis=0)
img_data = image_array / 255.0  # Normalize the image to [0, 1]

# Predict using the model
prediction = saved_model.predict(img_data)

# Assuming class names are: ['Normal', 'Benign', 'Malignant']
predicted_class = np.argmax(prediction)

if predicted_class == 'Normal':
    print('The ultrasound appears normal.')
elif predicted_class == 'Benign':
    print('The ultrasound shows benign signs.')
else:
    print('The ultrasound shows malignant signs indicative of breast cancer.')

print(f'Prediction Probabilities: {prediction}')


# In[13]:


# Load the trained model
saved_model = load_model('./baseline_model.h5')

# Image path
image_path = 'benign.jpeg'  # ext can be any image extension

# Load and preprocess the image in grayscale
img = image.load_img(image_path, color_mode='grayscale', target_size=(224, 224))
image_array = image.img_to_array(img)
image_array = np.expand_dims(image_array, axis=0)
img_data = image_array / 255.0  # Normalize the image to [0, 1]

# Predict using the model
prediction = saved_model.predict(img_data)

# Assuming class names are: ['Normal', 'Benign', 'Malignant']
predicted_class = np.argmax(prediction)

if predicted_class == 0:  # Normal
    print('The ultrasound appears normal.')
elif predicted_class == 1:  # Benign
    print('The ultrasound shows benign signs.')
else:  # Malignant
    print('The ultrasound shows malignant signs indicative of breast cancer.')

print(f'Prediction Probabilities: {prediction}')


# In[10]:


from PIL import ImageOps

# Image path
image_path = 'sample1.png'  # Replace with path to your image

# Load and preprocess the image
with open(image_path, "rb") as f:
    with Image.open(f) as img:
        # Convert to grayscale
        img = ImageOps.grayscale(img)
        # Resize to expected input size
        img = img.resize((224, 224))
        # Convert image to array and preprocess
        image_array = np.array(img).reshape(1, 224, 224, 1)
        img_data = image_array / 255.0  # Normalize the image to [0, 1]

# Predict using the model
prediction = saved_model.predict(img_data)

# Determine the predicted class
predicted_class = np.argmax(prediction)

if predicted_class == 'Normal':
    print('The ultrasound appears normal.')
elif predicted_class == 'Benign':
    print('The ultrasound shows benign signs.')
else:
    print('The ultrasound shows malignant signs indicative of breast cancer.')

print(f'Prediction Probabilities: {prediction}')


# In[ ]:




