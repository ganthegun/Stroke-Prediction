import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.sidebar.header('User Input Features')
age = st.sidebar.slider('Age (Year)', 0, 100, 50)
avg_glucose_level = st.sidebar.slider('Average Glucose Level (mg/dL)', 50, 300, 200)
bmi = st.sidebar.slider('BMI (kg/m²)', 0, 100, 20)
work_type = st.sidebar.selectbox('Work Type (Goverment Job)', ('Yes', 'No'))
smoking_status = st.sidebar.selectbox('Smoking Status (Unknown)', ('Yes', 'No'))

data = {
    'age': age,  
    'avg_glucose_level': avg_glucose_level,  
    'bmi': bmi,  
    'work_type_Govt_job': work_type,
    'smoking_status_Unknown': smoking_status
    }

input_df = pd.DataFrame(data, index=[0])
input_df['work_type_Govt_job'] = input_df['work_type_Govt_job'].map({'Yes': 1, 'No': 0})
input_df['smoking_status_Unknown'] = input_df['smoking_status_Unknown'].map({'Yes': 1, 'No': 0})

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

cols = ['age', 'avg_glucose_level', 'bmi']
ss = StandardScaler().fit(df[cols])
input_df[cols] = ss.transform(input_df[cols])

load_clf = pickle.load(open('model.pkl', 'rb'))
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.write("""
# Stroke Prediction App
This app predicts **Stroke**.\n
Data obtained from the [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
""")

st.subheader('User Input features')

st.write(data)

st.subheader('Prediction')

st.write('Stroke' if prediction[0] == 1 else 'No Stroke')

st.subheader('Prediction Probability')

st.write(prediction_proba)