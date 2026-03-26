import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- LOAD MODELS ----------------
clf_model = joblib.load('xgboost_classifier.pkl')
reg_model = joblib.load('XGBRegressor.pkl')

# ---------------- UI ----------------
st.title("EMI Eligibility & EMI Prediction App")
st.write("Enter financial details below:")

monthly_salary = st.number_input("Monthly Salary", min_value=0.0)
current_emi = st.number_input("Current EMI Amount", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0)
bank_balance = st.number_input("Bank Balance", min_value=0.0)
emergency_fund = st.number_input("Emergency Fund", min_value=0.0)
dependents = st.number_input("Number of Dependents", min_value=0)
monthly_rent = st.number_input("Monthly Rent", min_value=0.0)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    # Avoid division by zero
    salary_safe = monthly_salary if monthly_salary != 0 else 1

    # Feature engineering
    debt_to_income = current_emi / salary_safe
    total_expenses = monthly_rent
    expense_to_income = total_expenses / salary_safe
    total_savings = bank_balance + emergency_fund
    savings_ratio = total_savings / salary_safe
    risk_score = (expense_to_income + debt_to_income) - savings_ratio

    # ---------------- FULL FEATURE LIST ----------------
    columns = [
        'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size',
        'dependents', 'school_fees', 'college_fees', 'travel_expenses',
        'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount',
        'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
        'requested_tenure', 'debt_to_income', 'total_expenses',
        'expense_to_income', 'total_savings', 'savings_ratio', 'risk_score',
        'age_27.0', 'age_28.0', 'age_31.0', 'age_32.0', 'age_33.0', 'age_37.0',
        'age_38.0', 'age_39.0', 'age_47.0', 'age_48.0', 'age_49.0', 'age_57.0',
        'age_58.0', 'age_59.0', 'age_26', 'age_26.0', 'age_27', 'age_28',
        'age_31', 'age_32', 'age_32.0.0', 'age_33', 'age_37', 'age_38',
        'age_38.0.0', 'age_39', 'age_47', 'age_48', 'age_49', 'age_57',
        'age_58', 'age_58.0.0', 'age_59', 'gender_FEMALE', 'gender_Female',
        'gender_M', 'gender_MALE', 'gender_Male', 'gender_female',
        'gender_male', 'marital_status_Single', 'education_High School',
        'education_Post Graduate', 'education_Professional',
        'employment_type_Private', 'employment_type_Self-employed',
        'company_type_MNC', 'company_type_Mid-size', 'company_type_Small',
        'company_type_Startup', 'house_type_Own', 'house_type_Rented',
        'existing_loans_Yes', 'emi_scenario_Education EMI',
        'emi_scenario_Home Appliances EMI', 'emi_scenario_Personal Loan EMI',
        'emi_scenario_Vehicle EMI'
    ]

    # Create empty input
    input_dict = dict.fromkeys(columns, 0)

    # Fill user inputs
    input_dict['monthly_salary'] = monthly_salary
    input_dict['current_emi_amount'] = current_emi
    input_dict['credit_score'] = credit_score
    input_dict['bank_balance'] = bank_balance
    input_dict['emergency_fund'] = emergency_fund
    input_dict['dependents'] = dependents
    input_dict['monthly_rent'] = monthly_rent

    # Engineered features
    input_dict['debt_to_income'] = debt_to_income
    input_dict['total_expenses'] = total_expenses
    input_dict['expense_to_income'] = expense_to_income
    input_dict['total_savings'] = total_savings
    input_dict['savings_ratio'] = savings_ratio
    input_dict['risk_score'] = risk_score

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # ---------------- PREDICTIONS ----------------
    emi_pred = reg_model.predict(input_df)[0]
    eligibility = clf_model.predict(input_df)[0]
    label_map = {
    0: "Eligible",
    1: "High_Risk",
    2: "Not_Eligible"
    }

    # ---------------- OUTPUT ----------------
    st.subheader("Results:")
    st.write("Predicted EMI:", emi_pred)
    st.write("Eligibility:", label_map.get(eligibility, eligibility))

    st.success("Prediction completed")
