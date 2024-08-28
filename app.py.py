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
input_data = np.array([[int(permission_features[feature]) for feature in permission_features]])

# Debugging: Print the shape of the input data
st.write("Input data shape:", input_data.shape)

# Prediction
if st.button("Predict"):
    try:
        prediction = predict_result(input_data)
        st.write("Prediction Result:", "Malicious" if prediction[0][0] > 0.5 else "Benign")
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        st.error("Please ensure that the input data shape matches the model's expected input shape.")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info("This app classifies Android applications as malicious or benign based on the permissions they request. "
                "Select the permissions that your app requires, and click 'Predict' to see the result.")
