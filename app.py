import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
from dotenv import load_dotenv


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Automatic Ticket Classification",
    layout="centered"
)


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# Cache Model & Artifacts (Load Once)
# --------------------------------------------------
@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model("best_lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder


# --------------------------------------------------
# Cache Gemini Model
# --------------------------------------------------
@st.cache_resource
def load_gemini_model():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")


final_model, tokenizer, label_encoder = load_model_and_artifacts()
gemini_model = load_gemini_model()

MAX_LEN = 102


# --------------------------------------------------
# Text Cleaning (Same as Training)
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\sáéíóúäëïöüáàâèéêëìíîïòóôùúûñçß]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --------------------------------------------------
# Queue Prediction
# --------------------------------------------------
def predict_queue(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred = final_model.predict(padded, verbose=0)
    return label_encoder.inverse_transform([np.argmax(pred)])[0]


# --------------------------------------------------
# Gemini Reply (Fast + Retry + Fallback)
# --------------------------------------------------
def generate_gemini_reply(ticket_text, predicted_queue):
    prompt = f"""
You are a customer support assistant.
Write a short, polite acknowledgement (max 4 sentences).

Ticket:
{ticket_text}

Queue:
{predicted_queue}

Do not include names or signatures.
"""

    for _ in range(2):  # one retry
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 120
                }
            )
            return response.text.strip()
        except Exception:
            continue

    return (
        "Thank you for contacting our support team. "
        "Your request has been received and routed to the appropriate department. "
        "Our team will get back to you shortly."
    )


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("Automatic Ticket Classification & Reply System")

# Session State Initialization
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""
if "predicted_queue" not in st.session_state:
    st.session_state.predicted_queue = None
if "reply" not in st.session_state:
    st.session_state.reply = None


ticket_text = st.text_area(
    "Enter customer ticket text:",
    height=180,
    placeholder="Type the customer issue here...",
    value=st.session_state.ticket_text
)

col1, col2 = st.columns(2)

with col1:
    classify_clicked = st.button("Classify & Generate Reply")

with col2:
    clear_clicked = st.button("Clear")


# Clear Button Logic
if clear_clicked:
    st.session_state.ticket_text = ""
    st.session_state.predicted_queue = None
    st.session_state.reply = None
    st.experimental_rerun()


# Classification Logic
if classify_clicked:
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        st.session_state.ticket_text = ticket_text
        with st.spinner("Classifying ticket and generating response..."):
            st.session_state.predicted_queue = predict_queue(ticket_text)
            st.session_state.reply = generate_gemini_reply(
                ticket_text, st.session_state.predicted_queue
            )


# Output Display
if st.session_state.predicted_queue:
    st.subheader("Predicted Queue")
    st.success(st.session_state.predicted_queue)

if st.session_state.reply:
    st.subheader("Automated Reply")
    st.write(st.session_state.reply)
