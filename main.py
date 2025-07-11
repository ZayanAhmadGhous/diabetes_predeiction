import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model (inside app for compatibility)
model = RandomForestClassifier()
model.fit(X, y)

# App title
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🧪 Diabetes Prediction App")
st.write("Enter your health information to predict the likelihood of diabetes.")

# Sidebar Inputs
st.sidebar.header("Input Features")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 79)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 33)

# Input for prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("🔴 High risk of Diabetes")
    else:
        st.success("🟢 Low risk of Diabetes")

    st.write(f"Prediction Confidence: {round(np.max(prediction_proba)*100, 2)}%")

# Footer
st.markdown("""\n---\nBuilt by Zayan Ahmad Ghous using Streamlit and Scikit-learn\n""")
