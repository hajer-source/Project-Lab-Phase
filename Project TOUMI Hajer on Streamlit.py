import numpy as np
import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import zip_longest

# Set the page configuration
st.set_page_config(
    layout="wide",                    # "centered" or "wide"
    initial_sidebar_state="expanded")  # "collapsed", "expanded", or "auto"



st.title('''
            :red[CREDIT SCORING CLASSIFICATION]''')

st.markdown(''':blue[This application was tried on a dataset from Kaggle]''')

st.markdown(''':blue[First of all, fill out this form then you will see your score classification]''')

age = st.number_input("Enter your age:",min_value= 18, step= 1 )

annual_income = st.number_input("Enter your annual income:", min_value=0.0, step=100.0)

monthly_inhand_salary = st.number_input("Enter your monthly in hand income:", min_value=0.0, step=100.0)

total_emi_per_month = st.number_input("Enter your monthly credit installments:",min_value=10.0, step=50.0 )

num_bank_accounts = st.number_input("Enter the number of your bank accounts:", min_value=1, step= 1)

num_credit_card = st.number_input("Enter the number of your credit cards:", min_value=0, step= 1)

interest_rate = st.number_input("Enter the interest rate:", min_value= 1.00, step= 0.01)

num_of_loan = st.number_input("Enter the number of loans that you have:", min_value=0, step= 1)

delay_from_due_date = st.number_input("Enter the number of dalays from due dates:",min_value=0, step= 1)

num_of_delayed_payment = st.number_input("Enter the amount of dalayed payments:", min_value=0, step= 1)

changed_credit_limit = st.number_input("Enter the number of changed credit limits:", min_value= 0.00, step= 0.01)

num_credit_inquiries = st.number_input("Enter the number of your credit inquiries:", min_value=0, step= 1)

outstanding_debt = st.number_input("Enter the amount of your outstanding debt:", min_value= 0.00, step= 50.00)

credit_utilization_ratio = st.number_input("Enter your credit utilization ratio:", min_value= 0.00, step= 50.00)

credit_history_age = st.number_input("Enter your credit history age:", min_value=0, step= 1)

amount_invested_monthly = st.number_input("Enter your monthly invested amount:", min_value= 0.00, step= 50.00)

monthly_balance = st.number_input("Enter the amount of your monthly balance:", min_value= 0.00, step= 50.00)

st.text("Select the types of the loans that you already have had")

# Multiple checkboxes for types of loans
auto_loan = st.checkbox("Auto loan")
credit_builder_loan= st.checkbox("Credit_builder_loan")
debt_consolidation_loan= st.checkbox("Debt_consolidation_loan")
home_equity_loan = st.checkbox("Home_equity_loan")
mortgage_loan = st.checkbox("Mortgage_loan")
no_loan = st.checkbox("No_loan")
not_specified = st.checkbox("Not_specified")
payday_loan = st.checkbox("Payday_loan")
personal_loan = st.checkbox("Personal_loan")
student_loan = st.checkbox("Student_loan")

occupations = [
               'Scientist', 
               'Teacher', 
               'Engineer', 
               'Entrepreneur',
               'Developer', 
               'Lawyer',
               'Media_Manager',
               'Doctor',
               'Journalist',
               'Manager', 
               'Accountant', 
               'Musician',
               'Mechanic', 
               'Writer', 
               'Architect']

occupation_encoded = st.selectbox("Select your occupation: ", occupations)

creditmix =[
            'Standard',
            'Good',
            'Bad']

credit_mix = st.selectbox("Select your credit_mix: ",creditmix)

payment_behaviour = ['High_spent_Small_value_payments',
                     'Low_spent_Large_value_payments',
                     'Low_spent_Medium_value_payments',
                     'Low_spent_Small_value_payments',
                     'High_spent_Medium_value_payments',
                     'High_spent_Large_value_payments']

payment_behaviour_encoded = st.selectbox("Select your credit_payment behaviour: ", payment_behaviour)

loaded_le = joblib.load('le.pkl')
occupation_encoded = loaded_le.transform([occupation_encoded])
loaded_le.fit([
              'Scientist', 
               'Teacher', 
               'Engineer', 
               'Entrepreneur',
               'Developer', 
               'Lawyer',
               'Media_Manager',
               'Doctor',
               'Journalist',
               'Manager', 
               'Accountant', 
               'Musician',
               'Mechanic', 
               'Writer', 
               'Architect'])

loaded_le1 = joblib.load('le1.pkl')
credit_mix_encoded = loaded_le1.transform([credit_mix])
loaded_le1.fit(['Standard',
            'Good',
            'Bad'])

loaded_le2 = joblib.load('le2.pkl')
payment_behaviour_encoded = loaded_le2.transform(payment_behaviour)
loaded_le2.fit(['High_spent_Small_value_payments',
                     'Low_spent_Large_value_payments',
                     'Low_spent_Medium_value_payments',
                     'Low_spent_Small_value_payments',
                     'High_spent_Medium_value_payments',
                     'High_spent_Large_value_payments'])


# Load the scaler
scaler = joblib.load('scaler.pkl')


features_to_scale = [
        annual_income, monthly_inhand_salary, total_emi_per_month,
       num_bank_accounts, num_credit_card, interest_rate, num_of_loan,
       delay_from_due_date, num_of_delayed_payment, changed_credit_limit,
       num_credit_inquiries, outstanding_debt, credit_utilization_ratio,
       credit_history_age, amount_invested_monthly, monthly_balance]

# Check lengths of iterable features (skip scalar values)
feature_to_scale_lengths = []
for feature in features_to_scale:
    if hasattr(feature, '__len__'):  # Check if feature is iterable (has length)
        feature_to_scale_lengths.append(len(feature))
    else:
        feature_to_scale_lengths.append(1)  # If it's scalar, consider length as 1

print(f"feature lengths: {feature_to_scale_lengths}")

# Now, let's ensure that all features have the same length (pad shorter ones)
max_len = max(feature_to_scale_lengths)
data_to_scale_padded = [
    feature if hasattr(feature, '__len__') else [feature] * max_len  # Handle scalar by repeating it
    for feature in features_to_scale
]

# Use zip_longest to pad and transpose
data_padded = list(zip_longest(*data_to_scale_padded, fillvalue=0))

data_scaled = scaler.fit_transform(data_padded)
# Convert to numpy array
data = np.array(data_scaled)

# Reshape data for scaling (ensure it's 2D)
data = data.reshape(1, -1)
  # Adjust this depending on actual needs

payment_behaviour_encoded = np.array([1, 2, 3, 4, 5, 6])  # This is an array with shape (6,)

# Reshape 'payment_behaviour_encoded' properly
if isinstance(payment_behaviour_encoded, np.ndarray):
    if payment_behaviour_encoded.ndim > 1:  # If it's more than 1D, flatten it to (6,)
        payment_behaviour_encoded = payment_behaviour_encoded.flatten()
    if payment_behaviour_encoded.shape[0] == 1:
        payment_behaviour_encoded = payment_behaviour_encoded.reshape(-1, 1)  # If it's a scalar or single value, reshape it to (1, 1)
    elif payment_behaviour_encoded.shape[0] == 6:
        payment_behaviour_encoded = payment_behaviour_encoded[0]  # Just take the first element (if that's your intention)


input_vector_reshaped = data.copy() 
print("Shape of input_vector_reshaped before inserting and concatenation:", input_vector_reshaped.shape)
print(f"Shape of input_vector_reshaped before inserting: {input_vector_reshaped.shape}")



# Convert boolean variables to integers (0 or 1)
auto_loan = int(auto_loan)
mortgage_loan = int(mortgage_loan)

# Insert each feature one by one (convert scalars/booleans to numpy arrays)
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(age).reshape(-1, 1)])  # Insert 'age'
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(auto_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(credit_builder_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(debt_consolidation_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(home_equity_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(mortgage_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(no_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(not_specified).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(payday_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(personal_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(student_loan).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(occupation_encoded).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, np.array(credit_mix_encoded).reshape(-1, 1)])
input_vector_reshaped = np.hstack([input_vector_reshaped, payment_behaviour_encoded.reshape(1, 1)])

print(f"Reshaped input data shape: {input_vector_reshaped.shape}")

print(f"payment_behaviour_encoded shape is :{payment_behaviour_encoded.shape}")

# Ensure that input_vector_reshaped is now assigned to input_data

print(f"Final input data for prediction: {input_vector_reshaped.shape}")

# Load the trained model (make sure you save the model after training)
model = joblib.load('random_forest_model.joblib')

# Add a button to trigger the prediction
if st.button('Predict'):
    # Ensure input_data is defined (replace with actual input data creation process)
    if input_vector_reshaped is not None and len(input_vector_reshaped) > 0:
        
        # Reshape the input data before making the prediction
        if input_vector_reshaped.ndim == 1:
            input_data = input_vector_reshaped.reshape(1, -1)  # Reshape to 2D if it's a 1D array

        print(f"Reshaped input_data: {input_vector_reshaped.shape}")  # Debug print to verify the shape
        
        try:
                # Make the prediction using the model
                prediction = model.predict(input_vector_reshaped)
                
                # Show the result
                st.write(f"Prediction: {prediction[0]}")  # Assuming the model's prediction output is a single value (e.g., 0 or 1)
                
                if prediction[0] == 0:
                    st.write("The model predicts class 0.")
                elif prediction[0] == 1:
                    st.write("The model predicts class 1.")
                else:
                    st.write("The model predicts a class 2.")
        except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error(f"Input data shape mismatch: expected {scaler.scale_.shape[0]} features, got {input_vector_reshaped.shape[1]} features.")
    else:
        st.error("Input data is empty or not defined.")

#I would like to use an image according to this application

import streamlit as st

image_path = r'C:\\Users\\Hajer\\Desktop\\STR\\1.jpg'

st.image(image_path, caption="Credit scoring is important, try and see what is your score class is.", use_container_width=True)