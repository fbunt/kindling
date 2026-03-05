import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

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
    st.session_state.client = None
    st.session_state.chat_session = None
    st.session_state.current_model = None

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        try:
            client = genai.Client(api_key=env_key)
            st.session_state.models = list_models(client)
            st.session_state.api_key = env_key
            st.session_state.client = client
        except Exception:
            pass

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
            st.session_state.client = client
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
    uploaded_image = st.file_uploader(
        "Upload an image for your next message",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Ready to send")

    if st.button("Logout"):
        st.session_state.api_key = None
        st.session_state.models = []
        st.session_state.messages = []
        st.session_state.client = None
        st.session_state.chat_session = None
        st.session_state.current_model = None
        st.rerun()

# Create or recreate chat session when model changes
if st.session_state.current_model != model:
    st.session_state.chat_session = st.session_state.client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    st.session_state.current_model = model
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image_bytes" in msg:
            st.image(msg["image_bytes"], width=300)

if prompt := st.chat_input("Type a message..."):
    with st.chat_message("user"):
        st.write(prompt)
        if uploaded_image:
            st.image(uploaded_image, width=300)

    user_msg = {"role": "user", "content": prompt}
    payload = [prompt]
    if uploaded_image:
        image_bytes = uploaded_image.getvalue()
        payload.insert(0, types.Part(
            inline_data=types.Blob(
                mime_type=uploaded_image.type,
                data=image_bytes,
            )
        ))
        user_msg["image_bytes"] = image_bytes
        user_msg["image_mime"] = uploaded_image.type
    st.session_state.messages.append(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_session.send_message(payload)
                text = response.text
                st.write(text)
                st.session_state.messages.append({"role": "assistant", "content": text})
            except Exception as e:
                st.error(str(e))
