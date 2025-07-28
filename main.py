#\Minicinda\conda\ContextAwareChatbot\contextawarechatbot

import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from rag_pipeline import load_vector_store  # remove 'app.' if running inside /app

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit UI setup

st.set_page_config(page_title="🧠 Contextual Chatbot with RAG", layout="centered")
st.title("🧠 Contextual Chatbot with RAG")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User query input
query = st.text_input("Ask a question:")

if query: 
    # Load the vector store from PDF
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()

    #  Set up memory to remember conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Load a local model via HuggingFace transformers pipeline
    local_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",      
        max_length=512,
        temperature=0.5
    )

    llm = HuggingFacePipeline(pipeline=local_pipeline)

    # Build the RAG chatbot chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    # Run the chain with user query
    response = chain.run(query)

    #  Append to chat history
    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")
