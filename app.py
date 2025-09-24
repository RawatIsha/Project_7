import streamlit as st
import pickle
from tensorflow import keras
import tensorflow as tf  # needed for Dataset

# Loading model path from pickle
with open("/Users/ranjitsingh/Documents/Deep Learning/Test_Project/model_path.pkl", "rb") as f:
    model_path = pickle.load(f)

# Load the actual model
model = keras.models.load_model(model_path)

# Streamlit UI
st.title("Text Classification ")
st.write("Enter your tweet to check if the tweet is disaster or non-disaster.")

# User input
user_text = st.text_area("Enter your tweet:")

# Prediction
if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        # Wrap input in tf.data.Dataset and batch it
        sample_ds = tf.data.Dataset.from_tensor_slices([user_text]).batch(1)
        
        # Make prediction
        prediction = model.predict(sample_ds)[0][0]  # get the scalar value
        
        # Convert probability to label
        if prediction > 0.5:
            label = "Disaster Tweet"
        else:
            label = "Non-Disaster Tweet"
        
        st.success(f"Prediction: {label} (Probability: {prediction:.2f})")

     
