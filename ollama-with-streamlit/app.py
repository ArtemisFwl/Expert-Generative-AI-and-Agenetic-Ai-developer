import streamlit as st
from ollama import Client

# Create client
client = Client(host='http://localhost:11434')

st.title("Aman Deep - Ollama App")

prompt = st.text_area("Enter your question", height=200)

if st.button("Generate Response"):

    if prompt.strip() == "deepseek-r1:1.5b":
        st.warning("Please enter a prompt")

    else:
        with st.spinner("Thinking..."):
            response = client.chat(
                model="deepseek-r1:1.5b",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response["message"]["content"]
            st.success("Response:")
            st.write(answer)
