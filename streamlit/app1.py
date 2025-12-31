import streamlit as st

st.title("My First Streamlit APp created by Aman Deep")
st.write("Welcome, this app calculates square of a number")

st.header(" Select a number")
number=st.slider(" Pick a number", 0,100,25)

st.subheader("Result")

squared_number= number*number

st.write(f"The square of the number**(number)** is ** (squared_number)**")
