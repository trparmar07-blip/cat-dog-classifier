import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model(
    "cat_dog_classifier.keras"
)

# Title
st.title("Cat vs Dog Classifier")

st.write("Upload an image to classify Cat or Dog")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image
    img = img.resize((128,128))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Result
    if prediction[0][0] > 0.5:
        st.success("Prediction: DOG")
    else:
        st.success("Prediction: CAT")
