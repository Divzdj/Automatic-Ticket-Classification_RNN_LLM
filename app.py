import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
from dotenv import load_dotenv


# Streamlit Page Config

st.set_page_config(
    page_title="Automatic Ticket Classification",
    layout="centered"
)


# Load Model & Artifacts

MODEL_PATH = "best_lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"

final_model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 102


# Gemini Setup

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash"
)


# Text Cleaning (SAME AS TRAINING)

import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\sáéíóúäëïöüáàâèéêëìíîïòóôùúûñçß]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Prediction Function

def predict_queue(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred = final_model.predict(padded)
    label_index = np.argmax(pred)
    return label_encoder.inverse_transform([label_index])[0]


# Gemini Reply Function

def generate_gemini_reply(ticket_text, predicted_queue):
    prompt = f"""
You are a professional customer support assistant.

Customer Ticket:
\"\"\"{ticket_text}\"\"\"

Predicted Department:
{predicted_queue}

Write a polite, empathetic response.
Acknowledge the issue and assure resolution.
Do NOT include names, signatures, or placeholders.
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()



# Streamlit UI

st.title(" Automatic Ticket Classification & Reply System")

ticket_text = st.text_area(
    "Enter customer ticket text:",
    height=180,
    placeholder="Type the customer issue here..."
)

if st.button("Classify & Generate Reply"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Processing..."):
            predicted_queue = predict_queue(ticket_text)
            reply = generate_gemini_reply(ticket_text, predicted_queue)

        st.subheader("Predicted Queue")
        st.success(predicted_queue)

        st.subheader("Automated Reply")
        st.write(reply)
