import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# loading the trained model 
model_path = '../models/xgboost_fraud_model.pkl'
model = joblib.load(model_path)

#Title
st.title("💳 Card Guard – Credit Card Fraud Detection")

st.markdown("""
This app predicts whether a credit card transaction is **fraudulent or not** based on transaction details.
You can either:
- Fill in a single transaction manually
- Or upload a CSV file with multiple transactions (no labels!)        
""")

# sidebar for Input

st.sidebar.header("🧾 Single Transaction Input")

# Input fields

amt = st.sidebar.number_input("Transaction Amount ($)", min_value = 0.0 , value = 50.0)
age = st.sidebar.slider("Customer Age",18,90,30)
gender = st.sidebar.selectbox("Gender",("Male","Female"))
trans_hour = st.sidebar.slider("Transaction Hour(0-23)",0,23,12)
trans_day_of_week = st.sidebar.selectbox("Day of the Week",("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))

# Category one-hot encoded inputs

category_grocery_pos = st.sidebar.checkbox("Is Grocery POS ? ")
category_shopping_net = st.sidebar.checkbox("Is Shopping Net ?")
category_misc_net = st.sidebar.checkbox("Is Misc Net ?")

# Mapping Inputs
gender_encoded = 1 if gender == "Female" else 0 
day_map = {
    "Monday": 0,"Tuesday": 1,"Wednesday": 2, "Thursday": 3,
    "Friday": 4,"Saturday": 5 , "Sunday": 6
}
day_encoded = day_map[trans_day_of_week]

#Assemble features inn the correct order 
input_features = np.array([
    amt,
    age,
    gender_encoded,
    trans_hour,
    day_encoded,
    int(category_grocery_pos),
    int(category_shopping_net),
    int(category_misc_net)
]).reshape(1,-1)

#Prediction
if st.button("🔍 Detect Fraud"):
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    if prediction ==1 :
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Confidence: {1 - probability:.2%})")

# ============== Multi- Prediction Section =========
st.markdown("---")
st.subheader("📁 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload a CSV file with transaction data (no labels)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(df.head())

        # Sanity check: number of columns
        expected_cols = 8
        if df.shape[1] != expected_cols:
            st.error(f"❌ The uploaded file must contain exactly {expected_cols} columns.")
        else:
            if st.button("🔮 Predict Bulk Transactions"):
                preds = model.predict(df)
                probs = model.predict_proba(df)[:, 1]

                df_results = df.copy()
                df_results["Prediction"] = preds
                df_results["Confidence (%)"] = np.round(probs * 100, 2)
                df_results["Prediction_Label"] = df_results["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

                st.success("✅ Predictions completed!")
                st.dataframe(df_results)

                # Optionally allow user to download the results
                csv_download = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", csv_download, "fraud_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")

