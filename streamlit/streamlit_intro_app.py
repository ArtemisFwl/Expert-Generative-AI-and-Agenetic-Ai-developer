import streamlit as st
import pandas as pd
import numpy as np


## App title and description 

st.title("First Streeam lit app by Aman Deep")
st.write("This is a simple app to demonstrate the basic functionlities of streamlit")
st.write("Streamlit is an open-source app framework for Machine Learning and Data Science projects.")

# mInteractive widget in sidebar

st.sidebar.header("User input features")

#Text_input
user_name= st.sidebar.text_input("What is your name?","Aman Deep")

#Slider
age=st.sidebar.slider("Select your age",0,100,25)

#Select Box
favourite_colour= st.sidebar.selectbox("What is your favourite colour",["Blue", "Red","Green", "Yellow"])


# Main page content
st.header(f"Welcome {user_name}")
st.write(f"Your age is {age} and your favourite colour is {favourite_colour}.")

#Displaying Data

st.subheader("Here is some random data:")

#create a sample dataframe 

data= pd.DataFrame(np.random.random((5,5)), columns=["a", "b", "c", "d", "e"])

st.dataframe(data)
                   

# --- Checkbox to show/hide content ---
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)

# --- Button to trigger an action ---
if st.button("Say hello"):
    st.write("Hello there!")
else:
    st.write("Goodbye")