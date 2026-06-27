import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# trained in ../notebooks/01_diabetes_prediction.ipynb (logistic regression with
# median imputation and scaling inside the pipeline)
model_path = Path(__file__).parent / "diabetes_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Risk Checker")
st.caption("Estimates diabetes risk from eight routine health readings, trained on "
           "the Pima Indians dataset. This is a learning project, not medical advice.")

col1, col2 = st.columns(2)
pregnancies = col1.number_input("Pregnancies", 0, 20, 1)
glucose = col2.number_input("Glucose", 0, 300, 120)

col3, col4 = st.columns(2)
blood_pressure = col3.number_input("Blood pressure", 0, 200, 70)
skin = col4.number_input("Skin thickness", 0, 100, 20)

col5, col6 = st.columns(2)
insulin = col5.number_input("Insulin", 0, 900, 80)
bmi = col6.number_input("BMI", 0.0, 70.0, 28.0)

col7, col8 = st.columns(2)
pedigree = col7.number_input("Diabetes pedigree", 0.0, 3.0, 0.5)
age = col8.number_input("Age", 1, 120, 33)

if st.button("Check risk"):
    row = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age,
    }])

    # a 0 in these readings means "not recorded", same as in training
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if row[c].iloc[0] == 0:
            row[c] = np.nan

    risk = model.predict_proba(row)[0, 1]
    st.header(f"Estimated risk: {risk:.0%}")
    st.write("Higher risk, worth a check" if risk >= 0.35 else "Lower risk")

    # show which readings pushed the estimate up or down
    scaled = model.named_steps["scale"].transform(
        model.named_steps["impute"].transform(row))
    coefs = model.named_steps["model"].coef_[0]
    labels = ["Pregnancies", "Glucose", "Blood pressure", "Skin thickness",
              "Insulin", "BMI", "Diabetes pedigree", "Age"]
    drivers = sorted(zip(labels, scaled[0] * coefs), key=lambda x: abs(x[1]), reverse=True)

    st.subheader("What drove this")
    for name, value in drivers[:3]:
        st.write(f"{name}: {'raised' if value > 0 else 'lowered'} the estimate")
