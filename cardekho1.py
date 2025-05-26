import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# Load the saved models
try:
    encoder = pickle.load(open('C:/Users/User/onehot.pkl', 'rb'))
    
    scaler = pickle.load(open('C:/Users/User/scaler.pkl', 'rb'))
    model = pickle.load(open('C:/Users/User/car_price_model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Streamlit App
st.title(" Car Price Prediction App")
st.write("Enter the car details to predict its estimated price.")

# ---  User Inputs ---
year = st.slider("Year of Manufacture", min_value=2000, max_value=2025, value=2015, step=1)
owner_no = st.selectbox("Number of Owners", options=[1, 2, 3, 4, 5], index=0)
kms_driven = st.slider("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
mileage = st.slider("Mileage (km/l or km/kg)", min_value=1.0, max_value=50.0, value=18.0, step=0.1)
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8])

transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
insurance = st.selectbox("Insurance Validity", ["Third Party insurance", "Comprehensive", "Zero Dep",
       "Third Party", "2", "1", "Not Available"])
city = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
body_type = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "MUV", "Minivans", "Wagon"])

# --- 🛠 Preprocessing ---
# Convert categorical inputs into DataFrame (ensure column names match training data)

categorical_input = pd.DataFrame([[transmission, insurance, city, fuel_type, body_type]], 
                                 columns=["Transmission", "Insurance Validity", "city", "Fuel Type", "bt"])
categorical_input = categorical_input[encoder.feature_names_in_]
# Apply one-hot encoding (check for missing columns)
try:
    categorical_encoded = encoder.transform(categorical_input) # Convert to NumPy array
except Exception as e:
    st.error(f"Error encoding categorical inputs: {e}")
    st.stop()

# Scale numerical inputs
numerical_input = np.array([[year, owner_no, kms_driven, mileage, seats]])
try:
    numerical_scaled = scaler.transform(numerical_input)
except Exception as e:
    st.error(f"Error scaling numerical inputs: {e}")
    st.stop()

# Combine processed inputs
final_input = np.hstack((numerical_scaled, categorical_encoded))

# ---  Prediction ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(final_input)
        st.success(f"Predicted Car Price: ₹{round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
