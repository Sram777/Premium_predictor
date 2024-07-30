import streamlit as st
import pickle
import pandas as pd


# Load the trained model
with open('random_forest_reg.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler used for standardization
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define a function to preprocess inputs
def preprocess_input(weight, height, age, num_surgeries, binary_features):
    bmi = weight / ((height*0.01) ** 2)
    numeric_features = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'NumberOfMajorSurgeries': [num_surgeries]
    })
    standardized_numeric_features = scaler.transform(numeric_features)
    input_data = pd.DataFrame(standardized_numeric_features, columns=['Age', 'BMI', 'NumberOfMajorSurgeries'])
    for feature in binary_features:
        input_data[feature] = [st.session_state[feature]]
    return input_data

# Streamlit app
st.title('Premium Prediction App')

# Collect user inputs
Age = st.number_input('Age', min_value=0, format='%d')
Weight = st.number_input('Weight (kg)', min_value=0.0, format='%f')
Height = st.number_input('Height (cm)', min_value=0.0, format='%f')
NumberOfMajorSurgerieses = st.number_input('Number of Surgeries', min_value=0, format='%d')

st.subheader('Check the box if you have the following')
binary_features =['Diabetes','BloodPressureProblems','AnyTransplants','AnyChronicDiseases','KnownAllergies','HistoryOfCancerInFamily','NumberOfMajorSurgeries']
for feature in binary_features:
    st.checkbox(feature, key=feature)

if st.button('Predict Premium'):
    input_data = preprocess_input(Weight, Height, Age, NumberOfMajorSurgerieses, binary_features)
    premium_prediction = model.predict(input_data)
    st.write(f'The predicted premium is: {premium_prediction[0]:.2f}')
