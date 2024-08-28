#streamlit
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

#Load your trained model
model = load_model('android_permission.h5')

# Function to make predictions
def predict_result(input_data):
    # Assuming input_data is a numpy array matching the input shape of your model
    prediction = model.predict(input_data)
    return prediction

# Streamlit app title
st.title("Android Permissions Classification")

# Collecting user input for each permission (you can add more based on your dataset)
permission_features = {
    "android.permission.GET_ACCOUNTS": st.checkbox("GET ACCOUNTS"),
    "android.permission.READ_PROFILE": st.checkbox("READ PROFILE"),
    "android.permission.CAMERA": st.checkbox("CAMERA"),
    "android.permission.ACCESS_FINE_LOCATION": st.checkbox("ACCESS FINE LOCATION"),
    "android.permission.SEND_SMS": st.checkbox("SEND SMS"),
    # Add more checkboxes based on your dataset features...
}

# Convert user input to numpy array for prediction
input_data = np.array([[int(permission_features[feature]) for feature in permission_features]])

# Prediction
if st.button("Predict"):
    prediction = predict_result(input_data)
    st.write("Prediction Result:", "Malicious" if prediction[0][0] > 0.5 else "Benign")

# Streamlit app title
st.title("Android Permissions Classification")

# Collecting user input for each permission (you can add more based on your dataset)
permission_features = {
    "android.permission.GET_ACCOUNTS": st.checkbox("GET ACCOUNTS"),
    "android.permission.READ_PROFILE": st.checkbox("READ PROFILE"),
    "android.permission.CAMERA": st.checkbox("CAMERA"),
    "android.permission.ACCESS_FINE_LOCATION": st.checkbox("ACCESS FINE LOCATION"),
    "android.permission.SEND_SMS": st.checkbox("SEND SMS"),
    # Add more checkboxes based on your dataset features...
}

# Convert user input to numpy array for prediction
input_data = np.array([[int(permission_features[feature]) for feature in permission_features]])

# Prediction
if st.button("Predict"):
    prediction = predict_result(input_data)
    st.write("Prediction Result:", "Malicious" if prediction[0][0] > 0.5 else "Benign")
