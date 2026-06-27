import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# trained in ../notebooks/02_car_price_prediction.ipynb (one-hot encoding + linear
# regression on cleaned Quikr listings)
model_path = Path(__file__).parent / "car_price_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# the model's typical miss on unseen cars (mean absolute error from the notebook)
TYPICAL_ERROR = 150000

# pull the options the model was actually trained on
encoder = model.named_steps["columntransformer"].named_transformers_["onehotencoder"]
names = sorted(encoder.categories_[0])
companies = sorted(encoder.categories_[1])
fuels = sorted(encoder.categories_[2])

st.title("Used Car Price Estimator")
st.caption("Rough resale estimate from a model trained on Quikr listings.")

company = st.selectbox("Company", companies)
name = st.selectbox("Model", names)

col1, col2 = st.columns(2)
year = col1.number_input("Year", 1995, 2024, 2018)
kms = col2.number_input("Kilometres driven", 0, 500000, 40000, step=1000)
fuel = st.selectbox("Fuel type", fuels)

if st.button("Estimate price"):
    row = pd.DataFrame([{"name": name, "company": company, "year": year,
                         "kms_driven": kms, "fuel_type": fuel}])
    price = model.predict(row)[0]
    low = max(0, price - TYPICAL_ERROR)
    high = price + TYPICAL_ERROR
    st.header(f"About Rs {price:,.0f}")
    st.write(f"Usually within Rs {low:,.0f} and Rs {high:,.0f}")
