import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load("src/xgb_model.pkl")  # Your trained model
scaler = joblib.load("src/scaler.pkl")  # The scaler you saved during training

# All columns (including categorical ones like education, self_employed)
all_columns = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# Numerical columns for scaling
numerical_cols = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


# Function to make predictions
def predict(input_data):
    # Convert input data into a DataFrame (must match the training feature structure)
    input_df = pd.DataFrame([input_data], columns=all_columns)

    # Ensure 'education' and 'self_employed' are correctly encoded as 1 or 0
    input_df["education"] = input_df["education"].apply(
        lambda x: 1 if x == "Graduate" else 0
    )
    input_df["self_employed"] = input_df["self_employed"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    # Standardize only the numerical columns using the saved scaler
    input_scaled = scaler.transform(input_df[numerical_cols])

    # Preserve the categorical columns alongside the scaled numerical columns
    input_df[numerical_cols] = input_scaled
    print(input_df)
    # Make prediction
    prediction = model.predict(
        input_df[all_columns]
    )  # Pass all columns, including scaled ones
    print(prediction)
    return prediction


# Streamlit UI
st.title("Loan Prediction App")

# Collecting user input
no_of_dependents = st.number_input(
    "Number of Dependents", min_value=0, max_value=10, value=1
)
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Income per Annum (in INR)", min_value=0, value=500000)
loan_amount = st.number_input("Loan Amount (in INR)", min_value=0, value=100000)
loan_term = st.number_input(
    "Loan Term (in months)", min_value=12, max_value=360, value=60
)
cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900, value=650)
residential_assets_value = st.number_input(
    "Residential Assets Value (in INR)", min_value=0, value=2000000
)
commercial_assets_value = st.number_input(
    "Commercial Assets Value (in INR)", min_value=0, value=1000000
)
luxury_assets_value = st.number_input(
    "Luxury Assets Value (in INR)", min_value=0, value=500000
)
bank_asset_value = st.number_input(
    "Bank Asset Value (in INR)", min_value=0, value=1000000
)

# Prepare input data dictionary
input_data = {
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value,
}

# Convert categorical values to numerical
# input_data["education"] = 1 if input_data["education"] == "Graduate" else 0
# input_data["self_employed"] = 1 if input_data["self_employed"] == "Yes" else 0

# Make prediction
if st.button("Predict"):
    result = predict(input_data)
    st.write("Prediction: ", "Approved" if result == 1 else "Not Approved")
