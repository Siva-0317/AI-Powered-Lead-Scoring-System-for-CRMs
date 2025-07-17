import streamlit as st
import pandas as pd
import joblib
import pickle

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_lead_model.pkl")
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

st.title("AI-Powered CRM Lead Scoring")
st.write("Enter lead details below to get a conversion prediction:")

# Load columns from training data for dynamic form (load all rows for correct stats)
sample_df = pd.read_csv("data/cleaned_lead_scoring.csv")
input_cols = [col for col in sample_df.columns if col != "Converted"]

user_input = {}
for col in input_cols:
    if sample_df[col].dtype == object:
        options = list(encoders[col].classes_) if col in encoders else sorted(sample_df[col].dropna().unique())
        user_input[col] = st.selectbox(col, options)
    elif sample_df[col].dtype in [float, int]:
        min_val = float(sample_df[col].min())
        max_val = float(sample_df[col].max())
        mean_val = float(sample_df[col].mean())
        user_input[col] = st.number_input(
            col,
            value=mean_val,
            min_value=min_val,
            max_value=max_val,
            step=1.0 if sample_df[col].dtype == float else 1
        )
    else:
        user_input[col] = st.text_input(col, "")

if st.button("Predict Lead Conversion"):
    # Prepare input DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical columns
    for col in input_df.columns:
        if col in encoders:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except Exception:
                st.error(f"Invalid value for {col}. Please select a valid option.")
                st.stop()

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'Converted' if pred == 1 else 'Not Converted'}")
    st.info(f"Conversion Probability: {proba:.2f}")
    