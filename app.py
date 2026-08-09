import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("customer_churn_final_model.pkl")

st.title("Customer Churn Prediction Engine")
st.write("Predict customer churn probability using Machine Learning")

tenure = st.number_input("Tenure Months", min_value=0)

monthly = st.number_input("Monthly Charges")

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer", "Credit card"]
)

if st.button("Predict Churn"):

    data = pd.DataFrame({
    "Gender":["Male"],
    "Senior Citizen":[0],
    "Partner":["No"],
    "Dependents":["No"],
    "Phone Service":["Yes"],
    "Multiple Lines":["No"],
    "Internet Service":[internet],
    "Online Security":["No"],
    "Online Backup":["No"],
    "Device Protection":["No"],
    "Tech Support":["No"],
    "Streaming TV":["No"],
    "Streaming Movies":["No"],
    "Contract":[contract],
    "Paperless Billing":["Yes"],
    "Payment Method":[payment],
    "Tenure Months":[tenure],
    "Monthly Charges":[monthly],
    "Total Charges":[monthly*tenure],
    "TotalServices":[3],
    "TenureGroup":["1-2 years"],
    "ChargePerService":[monthly/3]
})
    st.write(data.columns)
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk Customer\nChurn Probability: {probability:.2%}")
    else:
        st.success(f"Low Risk Customer\nChurn Probability: {probability:.2%}")
