import streamlit as st
import pickle 
import numpy as np

#Load the saved Model 
model=pickle.load(open(r"D:\AI_ML\Expert-Generative-AI-and-Agenetic-Ai-developer\machine-learning\linear_regression_model.pkl","rb"))

st.title("Salary Prediction App")

st.write("This model predicts the salary based on years of experience using simple linear regresion SLR")


years_experience=st.number_input("Enter years of experience: ", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

if st.button("Predict Salary"):
    experience_input= np.array([[years_experience]])
    prediction= model.predict(experience_input)
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
    
st.write("The model was trained using the dataset of salaries and years of experience. Model built by Aman Deep")   