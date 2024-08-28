import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('android_permission.h5')

# Function to make predictions
def predict_result(input_data):
    prediction = model.predict(input_data)
    return prediction

# Define the feature names (as given)
features = [
    "android.permission.GET_ACCOUNTS", "com.sonyericsson.home.permission.BROADCAST_BADGE", 
    "android.permission.READ_PROFILE", "android.permission.MANAGE_ACCOUNTS", 
    "android.permission.WRITE_SYNC_SETTINGS", "android.permission.READ_EXTERNAL_STORAGE", 
    "android.permission.RECEIVE_SMS", "com.android.launcher.permission.READ_SETTINGS", 
    "android.permission.WRITE_SETTINGS", "com.google.android.providers.gsf.permission.READ_GSERVICES", 
    "android.permission.DOWNLOAD_WITHOUT_NOTIFICATION", "android.permission.GET_TASKS", 
    "android.permission.WRITE_EXTERNAL_STORAGE", "android.permission.RECORD_AUDIO", 
    "com.huawei.android.launcher.permission.CHANGE_BADGE", 
    "com.oppo.launcher.permission.READ_SETTINGS", "android.permission.CHANGE_NETWORK_STATE", 
    "com.android.launcher.permission.INSTALL_SHORTCUT", 
    "android.permission.READ_PHONE_STATE", "android.permission.CALL_PHONE", 
    "android.permission.WRITE_CONTACTS", "android.permission.READ_PHONE_STATE", 
    "com.samsung.android.providers.context.permission.WRITE_USE_APP_FEATURE_SURVEY", 
    "android.permission.MODIFY_AUDIO_SETTINGS", "android.permission.ACCESS_LOCATION_EXTRA_COMMANDS", 
    "android.permission.INTERNET", "android.permission.MOUNT_UNMOUNT_FILESYSTEMS", 
    "com.majeur.launcher.permission.UPDATE_BADGE", "android.permission.AUTHENTICATE_ACCOUNTS", 
    "com.htc.launcher.permission.READ_SETTINGS", "android.permission.ACCESS_WIFI_STATE", 
    "android.permission.FLASHLIGHT", "android.permission.READ_APP_BADGE", 
    "android.permission.USE_CREDENTIALS", "android.permission.CHANGE_CONFIGURATION", 
    "android.permission.READ_SYNC_SETTINGS", "android.permission.BROADCAST_STICKY", 
    "com.anddoes.launcher.permission.UPDATE_COUNT", "com.android.alarm.permission.SET_ALARM", 
    "com.google.android.c2dm.permission.RECEIVE", "android.permission.KILL_BACKGROUND_PROCESSES", 
    "com.sonymobile.home.permission.PROVIDER_INSERT_BADGE", 
    "com.sec.android.provider.badge.permission.READ", "android.permission.WRITE_CALENDAR", 
    "android.permission.SEND_SMS", "com.huawei.android.launcher.permission.WRITE_SETTINGS", 
    "android.permission.REQUEST_INSTALL_PACKAGES", "android.permission.SET_WALLPAPER_HINTS", 
    "android.permission.SET_WALLPAPER", "com.oppo.launcher.permission.WRITE_SETTINGS", 
    "android.permission.RESTART_PACKAGES", "me.everything.badger.permission.BADGE_COUNT_WRITE", 
    "android.permission.ACCESS_MOCK_LOCATION", "android.permission.ACCESS_COARSE_LOCATION", 
    "android.permission.READ_LOGS", "com.google.android.gms.permission.ACTIVITY_RECOGNITION", 
    "com.amazon.device.messaging.permission.RECEIVE", "android.permission.SYSTEM_ALERT_WINDOW", 
    "android.permission.DISABLE_KEYGUARD", "android.permission.USE_FINGERPRINT", 
    "me.everything.badger.permission.BADGE_COUNT_READ", "android.permission.CHANGE_WIFI_STATE", 
    "android.permission.READ_CONTACTS", "com.android.vending.BILLING", 
    "android.permission.READ_CALENDAR", "android.permission.RECEIVE_BOOT_COMPLETED", 
    "android.permission.WAKE_LOCK", "android.permission.ACCESS_FINE_LOCATION", 
    "android.permission.BLUETOOTH", "android.permission.CAMERA", 
    "com.android.vending.CHECK_LICENSE", "android.permission.FOREGROUND_SERVICE", 
    "android.permission.BLUETOOTH_ADMIN", "android.permission.VIBRATE", 
    "android.permission.NFC", "android.permission.RECEIVE_USER_PRESENT", 
    "android.permission.CLEAR_APP_CACHE", "com.android.launcher.permission.UNINSTALL_SHORTCUT", 
    "com.sec.android.iap.permission.BILLING", "com.htc.launcher.permission.UPDATE_SHORTCUT", 
    "com.sec.android.provider.badge.permission.WRITE", "android.permission.ACCESS_NETWORK_STATE", 
    "com.google.android.finsky.permission.BIND_GET_INSTALL_REFERRER_SERVICE", 
    "com.huawei.android.launcher.permission.READ_SETTINGS", "android.permission.READ_SMS", 
    "android.permission.PROCESS_INCOMING_CALLS"
]

# Collecting user input for each permission with unique keys
permission_features = {feature: st.checkbox(f"{feature}_{i}", key=f"{i}_{feature}") for i, feature in enumerate(features)}

# Convert user input to numpy array for prediction
input_data = np.array([[int(permission_features[feature]) for feature in features]])

# No need for padding if the input_data matches the required shape
required_features = len(features)
if input_data.shape[1] != required_features:
    st.write("Input data shape does not match model's required input shape.")
else:
    # Debug: Show the input data
    st.write("Input Data:", input_data)

    # Prediction
    if st.button("Predict"):
        try:
            prediction = predict_result(input_data)
            
            # Debug: Show the raw prediction output
            st.write("Raw Prediction Output:", prediction)
            
            # Assuming prediction[0][0] is the probability for "Malicious"
            result = "Malicious" if prediction[0][0] > 0.5 else "Benign"
            st.write("Prediction Result:", result)
        except Exception as e:
            st.write("An error occurred:", e)
