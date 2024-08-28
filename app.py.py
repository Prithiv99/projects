import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('android_permission.h5')

# Function to make predictions
def predict_result(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app title
st.title("Android Permissions Classification")

# Collecting user input for each permission (ensure no duplicate labels or keys)
permission_features = {
    "android.permission.GET_ACCOUNTS": st.checkbox("GET ACCOUNTS", key="get_accounts"),
    "android.permission.READ_PROFILE": st.checkbox("READ PROFILE", key="read_profile"),
    "android.permission.CAMERA": st.checkbox("CAMERA", key="camera"),
    "android.permission.ACCESS_FINE_LOCATION": st.checkbox("ACCESS FINE LOCATION", key="access_fine_location"),
    "android.permission.SEND_SMS": st.checkbox("SEND SMS", key="send_sms"),
    # Add more checkboxes with unique keys...
}

# Convert user input to numpy array for prediction
input_data = np.array([[int(permission_features[feature]) for feature in permission_features]])

# Prediction
if st.button("Predict"):
    prediction = predict_result(input_data)
    st.write("Prediction Result:", "Malicious" if prediction[0][0] > 0.5 else "Benign")
