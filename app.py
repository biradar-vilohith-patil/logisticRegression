import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Demographic Prediction", layout="centered")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with open(r'C:\Users\DELL\Desktop\LogisticReg\models\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'C:\Users\DELL\Desktop\LogisticReg\models\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Demographic Outcome Predictor")
st.markdown("Enter the target demographics below to predict the classification outcome.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)

with col2:
    salary = st.number_input("Estimated Salary ($)", min_value=0, max_value=500000, step=1000)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Generate Prediction"):
    features = np.array([[age, salary]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    st.markdown("### Results")
    if prediction[0] == 1:
        st.success("Positive Classification (1)")
    else:
        st.error("Negative Classification (0)")