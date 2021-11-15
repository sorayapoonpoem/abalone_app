import streamlit as st
import pandas as pd

st.write("""
# My First Web Application
Let's enjoy **data science** project!
""")

st.sidebar.header("User Input")
st.sidebar.subheader("Please enter your data:")


v_Sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])

v_Length = st.sidebar.slider('Length', min_value=0.0, max_value=1.0, value= 0.5)
v_Diameter = st.sidebar.slider('Diameter', min_value=0.0, max_value=1.0, value= 0.4)
v_Height = st.sidebar.slider('Height', min_value=0.0, max_value=1.0, value= 0.1)
v_Whole_weight = st.sidebar.slider('Whole Weight', min_value=0.0, max_value=3.0, value= 0.8)
v_Shucked_weight = st.sidebar.slider('Shucked Weight', min_value=0.0, max_value=2.0, value= 0.3)
v_Viscera_weight = st.sidebar.slider('Viscera Weight', min_value=0.0, max_value=1.0, value= 0.2)
v_Shell_weight = st.sidebar.slider('Shell Weight', min_value=0.0, max_value=2.0, value= 0.2)

# Change the value of sex to be {'M','F','I'} as stored in the trained dataset
if v_Sex == 'Male': 
    v_Sex = 'M'
elif v_Sex == 'Female': 
    v_Sex = 'F'
else: v_Sex = 'I'

# Store user input data in a dictionary
data = {
    'Sex': v_Sex,
    'Length': v_Length,
    'Diameter': v_Diameter,
    'Height': v_Height,
    'Whole_weight': v_Whole_weight,
    'Shucked_weight': v_Shucked_weight,
    'Viscera_weight': v_Viscera_weight,
    'Shell_weight': v_Shell_weight
    }

# Create a data frame from the above dictionary
df = pd.DataFrame(data, index=[0])

# Main Panel
st.header('Application of Abalone\'s Age Prediction:')
st.subheader('User Input:')

st.write(df)

# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)
# st.write(df)

#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['Sex']])
# st.write(cat_data)

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)

# Select only the first row (the user input data)
X_new = X_new[:1] 

#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])
#Show the X_new data frame on the screen
st.subheader('Pre-Processed Input:')
st.write(X_new)

import pickle
# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))

#Apply the normalization model to new data
X_new = load_nor.transform(X_new)

#Show the X_new data frame on the screen
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
#Show the prediction result on the screen
st.subheader('Prediction:')
st.write(prediction)