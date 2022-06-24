import pandas as pd
import joblib
import streamlit as st

st.markdown('# Selamat Datang di Aplikasi Prediksi Hewan')

# Function for user input
def user_input():
    Height = st.sidebar.slider('Height',50.0,150.0,100.0)
    Weight = st.sidebar.slider('Weight',20.0,100.0,40.0)

    data = {'Height':[Height],'Weight':[Weight]}

    features = pd.DataFrame(data)

    return features

# Catch the user input
input_df = user_input()

# Display the user input features
st.subheader('User Input Features')
print(['Height','Weight'])
st.write(input_df)

# Load and predict
model = joblib.load('logisticRegr.pkl')
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Display prediction result and probability
st.subheader('Prediction Result')
st.write(prediction)
st.subheader('Prediction Probability, 0 untuk Cat dan 1 untuk Dog')
st.write(prediction_proba)
