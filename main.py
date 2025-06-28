# C:\Minicinda\conda\Healthcare\Healthcare

import os
import json
import streamlit as st 
from groq import Groq
from dotenv import load_dotenv

st.set_page_config(
    page_title="Health Query ChatBot",
    page_icon="💊",
    layout="centered"
)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

client = Groq()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💊 Health Query ChatBot")

st.sidebar.title("Chat History")
for idx, message in enumerate(st.session_state.chat_history):
    role = "User" if message["role"] == "user" else "Assistant"
    st.sidebar.write(f"{role} ({idx + 1}): {message['content']}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])


user_prompt = st.chat_input("Ask any health-related question...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    messages = [
        {"role": "system", "content": "You are a helpful and professional health assistant. You provide accurate and easy-to-understand answers to health-related queries. You do not provide medical diagnoses or emergency advice, but offer general health guidance. Always recommend consulting a doctor for serious concerns."},
        *st.session_state.chat_history
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
