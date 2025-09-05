# Contents of ~/my_app/main_page.py
import streamlit as st

__version__ = "V0.2.1"

st.set_page_config(
    page_title="Open Chatbot Playground",
    page_icon="🛝🎈",
)
st.title("🛝🎈Open Chatbot Playground Web-UI App " + __version__)

st.sidebar.success("Select a page above.")
st.subheader("", divider='rainbow')
st.write(
    """
    ### Open Chatbot Playground 🛝🎈
    This app is an OpenRouter powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Ollama](https://ollama.com/)
    - [llamaIndex](https://docs.llamaindex.ai/en/stable/)

    💡 Note: 
    
        1. Openchat: OpenRouter API key required!
        2. LocalFileChat: No API key required! But Ollama must be installed and running locally!

    - [Source Code](https://github.com/showjim/openchat)

    Made by Jerry Zhou
    """
)

