import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image  # Import Image from PIL
import json

# Load the saved model and class indices
model = tf.keras.models.load_model('/content/drive/MyDrive/plant_disease_prediction.keras')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), 'Unknown')
    return predicted_class_index, predicted_class_name

# Streamlit interface
st.title("Plant Disease Classifier")
st.write("Upload an image of a plant leaf to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class_index, predicted_class_name = predict(image)  # Call the predict function
    st.write(f"Prediction Index: {predicted_class_index}")
    st.write(f"Prediction Name: {predicted_class_name}")
