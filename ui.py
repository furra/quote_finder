import os

from dotenv import load_dotenv
from uuid import uuid4

import streamlit as st

from workflow import initialize_graph, stream

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

# Set the app title
st.title("Quote Finder - an AI to look for Author and Book quotes")

# Sidebar for API key input
with st.sidebar:
    if groq_api_key is None:
        groq_api_key = st.text_input("Enter your Groq API Key", type="password")
        st.warning("Please enter your Groq API key to use the chatbot.")

if st.button('Clear history'):
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.conversation_id = str(uuid4())


# Initialize resources only if API key is provided
if groq_api_key:
    with st.spinner("Initializing chatbot..."):
        workflow = initialize_graph()
        st.success("Chatbot initialized successfully!", icon="🚀")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can give you quotes from authors or books!",
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Your question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


# Generate answer if API key is provided
if groq_api_key:
    if prompt:
        with st.spinner("Thinking..."):
            response = stream(workflow, prompt, st.session_state.conversation_id)
        with st.chat_message("assistant"):
            st.markdown(response[-1])
        st.session_state.messages.append({"role": "assistant", "content": response[-1]})
else:
    st.error("Please enter your Groq API key in the sidebar to use the chatbot.")

