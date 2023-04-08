!pip install -r requirements.txt

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array

from keras.models import load_model
# Load the saved model
model_path = "/home/kalyan/DataSets/animals/self_resnet50.h5"
model = load_model(model_path)
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
data_classes = ["antelope","bat","beaver","bobcat","buffalo","chihuahua","chimpanzee","collie","dalmatian","german+shepherd","grizzly+bear",
                "hippopotamus","horse","killer+whale","mole","moose","mouse","otter","ox","persian+cat","raccoon","rat","rhinoceros","seal",
                "siamese+cat","spider+monkey","squirrel","walrus","weasel","wolf"]
#start label_map with 1
label_map = {i: data_classes[i] for i in range(0, len(data_classes))}

# Define the Streamlit app
st.title('Classification')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = make_prediction(image)
    animal_class = np.argmax(prediction)
    st.write('Animal class:', animal_class-1)
    st.write('Animal name:', label_map[animal_class-1])
