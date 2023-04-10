import io
import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.utils import img_to_array
from google_drive_downloader import GoogleDriveDownloader as gdd

# Define the function to download a file from Google Drive
def download_file_from_google_drive(id, destination):
    gdd.download_file_from_google_drive(file_id=id, dest_path=destination)

# Define the function to load a TensorFlow model from a file path
def load_tf_model(file_path):
    with open(file_path, "rb") as f:
        model = tf.keras.models.load_model(f)
    return model

# Define the Streamlit app
st.title('Classification')

# Define the Google Drive file ID for the model
model_id = '1QoJdVNtdR3vYJEoFMgtSJ1ZkfpvUjmor'

# Define the file path to save the downloaded model
model_path = 'model.h5'

# Check if the model file exists, and download it if it doesn't
if not os.path.exists(model_path):
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        download_file_from_google_drive(model_id, model_path)

# Load the TensorFlow model
model = load_tf_model(model_path)

# Define the function to make a prediction
def make_prediction(image):
    # Preprocess the image
    image = image.resize((64,64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    
    # Make a prediction
    prediction = model.predict(image)
    
    return prediction

# Define the data classes
data_classes = ["antelope","bat","beaver","bobcat","buffalo","chihuahua","chimpanzee","collie","dalmatian","german+shepherd","grizzly+bear",
                "hippopotamus","horse","killer+whale","mole","moose","mouse","otter","ox","persian+cat","raccoon","rat","rhinoceros","seal",
                "siamese+cat","spider+monkey","squirrel","walrus","weasel","wolf"]

# Define the label map
label_map = {i: data_classes[i] for i in range(0, len(data_classes))}

# Define the file uploader and prediction display
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = make_prediction(image)
    animal_class = np.argmax(prediction)
    st.write('Animal class:', animal_class)
    st.write('Animal name:', label_map[animal_class])
