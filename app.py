import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
from dotenv import load_dotenv


# Streamlit Page Config
st.set_page_config(
    page_title="Automatic Ticket Classification",
    layout="centered"
)


# Load environment variables
load_dotenv()


# Cache Model & Artifacts
@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model("best_lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder


# Cache Gemini Model
@st.cache_resource
def load_gemini_model():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")


final_model, tokenizer, label_encoder = load_model_and_artifacts()
gemini_model = load_gemini_model()

MAX_LEN = 102


# Text Cleaning (Same as Training)
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
    pred = final_model.predict(padded, verbose=0)
    label_index = np.argmax(pred)
    return label_encoder.inverse_transform([label_index])[0]


# Gemini Reply Function (Timeout Safe)
def generate_gemini_reply(ticket_text, predicted_queue):
    prompt = f"""
You are a professional customer support assistant.
Write a short, polite acknowledgement (max 4 sentences).

Customer Ticket:
{ticket_text}

Predicted Department:
{predicted_queue}

Do not include names, signatures, or placeholders.
"""

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
        return (
            "Thank you for contacting our support team. "
            "Your request has been received and has been forwarded to the appropriate department. "
            "Our team will get back to you shortly."
        )


# Streamlit UI
st.title("Automatic Ticket Classification & Reply System")

ticket_text = st.text_area(
    "Enter customer ticket text:",
    height=180,
    placeholder="Type the customer issue here..."
)

if st.button("Classify & Generate Reply"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Classifying ticket and generating response..."):
            predicted_queue = predict_queue(ticket_text)
            reply = generate_gemini_reply(ticket_text, predicted_queue)

        st.subheader("Predicted Queue")
        st.success(predicted_queue)

        st.subheader("Automated Reply")
        st.write(reply)
