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

# Define permission features based on your dataset
permission_features = {
    "android.permission.GET_ACCOUNTS": st.checkbox("GET ACCOUNTS", key="get_accounts"),
    "android.permission.READ_PROFILE": st.checkbox("READ PROFILE", key="read_profile"),
    "android.permission.CAMERA": st.checkbox("CAMERA", key="camera"),
    "android.permission.ACCESS_FINE_LOCATION": st.checkbox("ACCESS FINE LOCATION", key="access_fine_location"),
    "android.permission.SEND_SMS": st.checkbox("SEND SMS", key="send_sms"),
    "android.permission.RECEIVE_SMS": st.checkbox("RECEIVE SMS", key="receive_sms"),
    "android.permission.READ_CONTACTS": st.checkbox("READ CONTACTS", key="read_contacts"),
    "android.permission.WRITE_EXTERNAL_STORAGE": st.checkbox("WRITE EXTERNAL STORAGE", key="write_external_storage"),
    "android.permission.READ_EXTERNAL_STORAGE": st.checkbox("READ EXTERNAL STORAGE", key="read_external_storage"),
    # Add more checkboxes as needed...
}

# Convert user input to numpy array for prediction
# Assuming you have 9 features from checkboxes
input_data = np.array([[int(permission_features[feature]) for feature in permission_features]])

# Padding the input data to match the required shape (add 77 zeros)
input_data_padded = np.pad(input_data, ((0, 0), (0, 77)), 'constant', constant_values=0)

# Use the padded input data for prediction
if st.button("Predict"):
    prediction = predict_result(input_data_padded)
    st.write("Prediction Result:", "Malicious" if prediction[0][0] > 0.5 else "Benign")
