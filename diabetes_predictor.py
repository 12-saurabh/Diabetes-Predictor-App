import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# Load dataset (Using sklearn's built-in)
@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return data

df = load_data()

# Prepare model
X = df.drop('Outcome', axis=1)
y = df['Outcome']

model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.slider("Glucose Level", 0, 200, 100)
bp = st.slider("Blood Pressure", 0, 140, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 600, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Age", 10, 100, 30)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)[0]
    st.success("Positive for Diabetes" if result == 1 else "Negative for Diabetes")

