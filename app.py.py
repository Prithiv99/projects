import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('android_permission.h5')

# Function to make predictions
def predict_result(input_data):
    prediction = model.predict(input_data)
    return prediction

# Collecting user input for each permission (example with 10 features)
permission_features = {
    "android.permission.GET_ACCOUNTS": st.checkbox("GET ACCOUNTS", key="get_accounts"),
    "android.permission.READ_PROFILE": st.checkbox("READ PROFILE", key="read_profile"),
    "android.permission.CAMERA": st.checkbox("CAMERA", key="camera"),
    "android.permission.ACCESS_FINE_LOCATION": st.checkbox("ACCESS FINE LOCATION", key="access_fine_location"),
    "android.permission.SEND_SMS": st.checkbox("SEND SMS", key="send_sms"),
    "android.permission.READ_CONTACTS": st.checkbox("READ CONTACTS", key="read_contacts"),
    "android.permission.WRITE_EXTERNAL_STORAGE": st.checkbox("WRITE EXTERNAL STORAGE", key="write_external_storage"),
    "android.permission.RECORD_AUDIO": st.checkbox("RECORD AUDIO", key="record_audio"),
    "android.permission.READ_SMS": st.checkbox("READ SMS", key="read_sms"),
    "android.permission.ACCESS_WIFI_STATE": st.checkbox("ACCESS WIFI STATE", key="access_wifi_state"),
}

# Convert user input to numpy array for prediction
input_data = np.array([[int(permission_features[feature]) for feature in permission_features]])

# Padding the input data to match the required shape (if needed)
required_features = 86  # Adjust this to match your model's input shape
if input_data.shape[1] < required_features:
    input_data = np.pad(input_data, ((0, 0), (0, required_features - input_data.shape[1])), 'constant', constant_values=0)

# Debug: Show the input data
st.write("Input Data:", input_data)

# Prediction
if st.button("Predict"):
    prediction = predict_result(input_data)
    
    # Debug: Show the raw prediction output
    st.write("Raw Prediction Output:", prediction)
    
    # Assuming prediction[0][0] is the probability for "Malicious"
    result = "Malicious" if prediction[0][0] > 0.5 else "Benign"
    st.write("Prediction Result:", result)
