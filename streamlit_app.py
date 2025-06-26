import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

st.write("""
# Stroke Prediction App

This app predicts the **Stroke**!

Data obtained from the [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
""")

st.sidebar.header('User Input Features')
age = st.sidebar.slider('Age (Year)', 0, 100, 50)
avg_glucose_level = st.sidebar.slider('Average Glucose Level (mg/dL)', 50, 300, 200)
bmi = st.sidebar.slider('BMI (kg/m²)', 0, 100, 20)
hypertension = st.sidebar.selectbox('Hypertension', ('Yes', 'No'))
heart_disease = st.sidebar.selectbox('Heart Disease', ('Yes', 'No'))
work_type_private = st.sidebar.selectbox('Work Type (Private only)', ('Yes', 'No')) == 'Yes'
smoking_status_formerly_smoked = st.sidebar.selectbox('Smoking Status (formerly smoked only)', ('Yes', 'No')) == 'Yes'

data = {
    'age': age,  
    'avg_glucose_level': avg_glucose_level,  
    'bmi': bmi,  
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'work_type_Private': 1 if work_type_private else 0,
    'smoking_status_formerly smoked': 1 if smoking_status_formerly_smoked else 0,
    }

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
ss = StandardScaler()
ss.fit(df[['age', 'avg_glucose_level', 'bmi']])
features_to_scale = [[data['age'], data['avg_glucose_level'], data['bmi']]]
scaled_features = ss.transform(features_to_scale)[0]
data['age'] = scaled_features[0]
data['avg_glucose_level'] = scaled_features[1]
data['bmi'] = scaled_features[2]


input_df = pd.DataFrame(data, index=[1])


# Displays the user input features
st.subheader('User Input features')
st.write(input_df)

# Process the input data to match how the model was trained
def prepare_input_for_model(input_data):
    # Load the model
    load_clf = pickle.load(open('stroke_rfc.pkl', 'rb'))
    
    # Get the expected feature names from the model
    expected_features = load_clf.feature_names_in_
    
    # Add any missing features to the input data with a default value of 0
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    
    # Reorder the columns to match the model's expected feature order
    input_data = input_data[expected_features]
    
    return load_clf, input_data

# Prepare data and get model
model, processed_data = prepare_input_for_model(input_df)

# Make prediction
try:
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    
    st.subheader('Prediction')
    st.write('Stroke' if prediction[0] == 1 else 'No Stroke')
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
except Exception as e:
    st.error(f"Prediction error: {e}")
    
    # Debugging info
    st.subheader("Debug Information")
    st.write("Model expected features:")
    if hasattr(model, 'feature_names_in_'):
        st.write(model.feature_names_in_)
    
    st.write("Features provided:")
    st.write(processed_data.columns.tolist())