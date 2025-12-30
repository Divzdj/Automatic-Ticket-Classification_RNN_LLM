import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Automatic Ticket Classification",
    layout="centered"
)

# ---------------- Load Environment ----------------
load_dotenv()

# ---------------- Cache Model & Artifacts ----------------
@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model("best_lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

@st.cache_resource
def load_gemini_model():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")

model, tokenizer, label_encoder = load_model_and_artifacts()
gemini_model = load_gemini_model()

MAX_LEN = 102

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- Prediction ----------------
def predict_queue(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    preds = model.predict(padded, verbose=0)
    return label_encoder.inverse_transform([np.argmax(preds)])[0]

# ---------------- Gemini Reply ----------------
def generate_gemini_reply(ticket_text, predicted_queue):
    prompt = f"""
You are a professional customer support assistant.

Customer ticket:
{ticket_text}

Predicted department:
{predicted_queue}

Write a polite, empathetic response acknowledging the issue
and assuring the customer that it will be handled appropriately.
Do not include names, signatures, or placeholders.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return (
            "Thank you for contacting us. "
            "Your concern has been forwarded to the appropriate department, "
            "and our team will get back to you as soon as possible."
        )

# ---------------- Session State Init ----------------
for key in ["ticket_text", "predicted_queue", "reply"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ---------------- Clear Callback ----------------
def clear_all():
    st.session_state["ticket_text"] = ""
    st.session_state["predicted_queue"] = ""
    st.session_state["reply"] = ""

# ---------------- UI ----------------
st.title("Automatic Ticket Classification & Reply System")

# ---- Form (ONLY classify here) ----
with st.form("ticket_form"):
    st.text_area(
        "Enter customer ticket text:",
        key="ticket_text",
        height=180
    )

    classify_btn = st.form_submit_button("Classify & Generate Reply")

# ---- Clear button OUTSIDE form ----
st.button("Clear", on_click=clear_all)

# ---------------- Actions ----------------
if classify_btn:
    if st.session_state.ticket_text.strip():
        with st.spinner("Processing..."):
            st.session_state.predicted_queue = predict_queue(
                st.session_state.ticket_text
            )
            st.session_state.reply = generate_gemini_reply(
                st.session_state.ticket_text,
                st.session_state.predicted_queue
            )
    else:
        st.warning("Please enter a ticket description.")

# ---------------- Output ----------------
if st.session_state.predicted_queue:
    st.subheader("Predicted Queue")
    st.success(st.session_state.predicted_queue)

if st.session_state.reply:
    st.subheader("Automated Reply")
    st.write(st.session_state.reply)
