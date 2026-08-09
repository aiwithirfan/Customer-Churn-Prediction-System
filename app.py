import streamlit as st
import joblib
import pandas as pd


# Load Model
model = joblib.load("customer_churn_final_model.pkl")


# Title
st.title("Customer Churn Prediction Engine")
st.write("Predict customer churn probability using Machine Learning")


# User Inputs

tenure = st.number_input(
    "Tenure Months",
    min_value=0,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


internet = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)



if st.button("Predict Churn"):


    # Feature Engineering (same as training)

    total_services = 3


    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 36:
        tenure_group = "Established"
    else:
        tenure_group = "Loyal"


    charge_per_service = monthly / (total_services + 1)



    # Input Data

    data = pd.DataFrame({

        "Gender": ["Male"],
        "Senior Citizen": ["No"],
        "Partner": ["Yes"],
        "Dependents": ["No"],

        "Tenure Months": [tenure],

        "Phone Service": ["Yes"],
        "Multiple Lines": ["No"],

        "Internet Service": [internet],

        "Online Security": ["No"],
        "Online Backup": ["No"],
        "Device Protection": ["No"],
        "Tech Support": ["No"],

        "Streaming TV": ["No"],
        "Streaming Movies": ["No"],

        "Contract": [contract],

        "Paperless Billing": ["Yes"],

        "Payment Method": [payment],

        "Monthly Charges": [monthly],

        "Total Charges": [str(monthly * tenure)],

        "TotalServices": [total_services],

        "TenureGroup": [tenure_group],

        "ChargePerService": [charge_per_service]

    })



    # Convert categorical columns

    cat_cols = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Total Charges",
        "TenureGroup"
    ]


    data[cat_cols] = data[cat_cols].astype(str)



    try:

        prediction = model.predict(data)

        probability = model.predict_proba(data)[0][1]


        if prediction[0] == 1:

            st.error(
                f"⚠️ High Risk Customer\n\nChurn Probability: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ Low Risk Customer\n\nChurn Probability: {probability:.2%}"
            )


    except Exception as e:

        st.error("Prediction failed")
        st.write(e)
