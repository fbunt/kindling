import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="natlangq", layout="centered")
st.title("natlangq")


def list_models(client):
    models = []
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            models.append(m.name)
    models.sort()
    return models


# --- Auth ---
if "api_key" not in st.session_state:
    st.session_state.api_key = None
    st.session_state.models = []
    st.session_state.messages = []

if not st.session_state.api_key:
    st.markdown("Enter your Google Gemini API key to get started.")
    with st.form("login"):
        api_key = st.text_input("Gemini API key", type="password")
        submitted = st.form_submit_button("Connect")
    if submitted and api_key:
        try:
            client = genai.Client(api_key=api_key)
            st.session_state.models = list_models(client)
            st.session_state.api_key = api_key
            st.rerun()
        except Exception as e:
            st.error(f"Invalid API key: {e}")
    st.stop()

# --- Chat UI ---
with st.sidebar:
    model = st.selectbox(
        "Model",
        st.session_state.models,
        index=(
            next(
                (i for i, m in enumerate(st.session_state.models) if "gemini-3.1-pro-preview" in m),
                0,
            )
        ),
    )
    if st.button("Logout"):
        st.session_state.api_key = None
        st.session_state.models = []
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = genai.Client(api_key=st.session_state.api_key)
    contents = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    with st.chat_message("assistant"):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            text = response.text
            st.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
        except Exception as e:
            st.error(str(e))
