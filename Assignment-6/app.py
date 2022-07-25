import streamlit as st
import pickle

st.title("Salary Prediction")

text = st.number_input("Enter Years of Experience")
text=int(text)
model = pickle.load(open('model.pkl','rb'))

if st.button('Predit Salary'):
    pred = model.predict([[text]])
    pred = "Your Salary Prediction is %.3f" %pred
    st.success(pred)
    