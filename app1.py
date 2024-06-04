import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
with open('Random.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title('Titanic Survival Prediction')

# Input fields for the user
Pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
Fare = st.number_input('Fare', min_value=0.0, step=0.1)
Age = st.number_input('Age', min_value=0, step=1)
Sex = st.selectbox('Sex', ['male', 'female'])
Embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
Family_member = st.number_input('Family Members (SibSp + Parch)', min_value=0, step=1)
Title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Mme', 'Ms', 'Sir', 'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona', 'Capt'])

# Preprocess inputs
Age_group_Child = Age_group_Teen = Age_group_Adult = Age_group_Middle_Age = Age_group_Old = 0
if Age < 13:
    Age_group_Child = 1
elif 13 <= Age < 20:
    Age_group_Teen = 1
elif 20 <= Age < 40:
    Age_group_Adult = 1
elif 40 <= Age < 60:
    Age_group_Middle_Age = 1
else:
    Age_group_Old = 1

Sex_female = Sex_male = 0
if Sex == 'female':
    Sex_female = 1
else:
    Sex_male = 1

Embarked_C = Embarked_Q = Embarked_S = 0
if Embarked == 'C':
    Embarked_C = 1
elif Embarked == 'Q':
    Embarked_Q = 1
else:
    Embarked_S = 1

# Encode title
title_encoder = LabelEncoder()
title_encoder.classes_ = np.array(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'])
Title_encoded = title_encoder.transform([Title])[0]

# Create the feature vector
features = np.array([[Pclass, Fare, Age_group_Child, Age_group_Teen, Age_group_Adult, Age_group_Middle_Age, Age_group_Old, Sex_female, Sex_male, Family_member, Title_encoded]])

# Predict
if st.button('Predict Survival'):
    prediction = model.predict(features)
    if prediction == 1:
        st.success('The passenger would have survived.')
    else:
        st.error('The passenger would not have survived.')
