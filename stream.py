import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_xgb_insurance_model.pkl')

# Function to predict insurance cost
def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'female' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title('Medical Insurance Cost Predictor')

# Input fields for user to enter their data
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', ('male', 'female'))
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=1)
smoker = st.selectbox('Smoker', ('yes', 'no'))
region = st.selectbox('Region', ('northeast', 'southeast', 'northwest', 'southwest'))

# Predict button
if st.button('Predict'):
    predicted_cost = predict_insurance_cost(age, sex, bmi, children, smoker, region)
    st.success(f"The predicted medical insurance cost is ${predicted_cost:.2f}")

