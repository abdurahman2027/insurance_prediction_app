import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("Insurance Charges Predictor")
st.write("Enter the details below to predict insurance charges of an individual.")

#Adding studnet information
st.sidebar.image("profile.jpg", width=180)
st.sidebar.markdown("**Student Name:** Abdu Rahman")
st.sidebar.markdown("**Student ID:** PIUS20230015")

# --- Load the saved model (pipeline) ---
def load_model():
    with open("insurance_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- User Inputs ---
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# --- Engineered Features ---
smoker_bin = 1 if smoker == "yes" else 0
age_bmi = age * bmi
age_smoker = age * smoker_bin
bmi_smoker = bmi * smoker_bin

# --- Prepare input row matching training columns ---
input_df = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'children': children,
    'smoker_bin': smoker_bin,
    'age_bmi': age_bmi,
    'age_smoker': age_smoker,
    'bmi_smoker': bmi_smoker,
    'sex': sex,
    'region': region,
    'smoker': smoker
}])

# --- Predict button ---
if st.button("Predict Insurance Charge"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Insurance Charge: ${prediction:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Please check inputs and model.")
        st.write(e)

