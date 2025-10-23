import streamlit as st
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- NLTK Data Download ---
# We do this once at the start
@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
load_nltk_data()

# --- Load All Artifacts ---
MODEL_PATH = 'deep_csat_text_model.h5'
TOKENIZER_PATH = 'tokenizer.json'

# Use st.cache_resource to load the model and tokenizer only once
@st.cache_resource
def load_app_artifacts():
    # 1. Load Model
    model = load_model(MODEL_PATH)
    
    # 2. Load Tokenizer
    with open(TOKENIZER_PATH) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    # 3. Load Preprocessing Utilities
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    max_length = 50  # Must be the same as in training!
    
    return model, tokenizer, stop_words, lemmatizer, max_length

# Load everything
model, tokenizer, stop_words, lemmatizer, max_length = load_app_artifacts()

# --- Text Preprocessing Function ---
# This function must be identical to the one used in training
def clean_text_for_prediction(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_text = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    return padded_text

# --- Streamlit App UI ---

st.title('DeepCSAT: Customer Satisfaction Prediction ')
st.write("This app uses a Deep Learning model to predict a CSAT score (1-5) based on a customer's text feedback.")

# User input
st.header("Enter Customer Feedback:")
user_input = st.text_area("Type the customer's remark, chat log, or review here:", height=150)

# Prediction button
if st.button('Predict CSAT Score'):
    if user_input:
        try:
            # 1. Preprocess the input
            processed_input = clean_text_for_prediction(user_input)
            
            # 2. Make Prediction
            prediction = model.predict(processed_input)
            predicted_score = float(prediction[0][0])
            
            # 3. Display the result
            st.header(f"Predicted CSAT Score: {predicted_score:.2f} / 5.0")
            
            # Use st.metric for a nice visual
            st.metric(label="Predicted Score", value=f"{predicted_score:.2f}")

            # Add a conditional emoji/message for context
            if predicted_score >= 4.5:
                st.success("This customer is very satisfied! 😊")
            elif predicted_score >= 3.5:
                st.info("This customer is neutral or mildly satisfied. 🙂")
            elif predicted_score >= 2.5:
                st.warning("This customer is dissatisfied. 😟")
            else:
                st.error("This customer is very unhappy. 😡 Action may be required.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some text to predict.")

st.sidebar.header("About This Project")
st.sidebar.info(
    "**Project Goal:** Develop a deep learning model to predict CSAT scores "
    "from customer interactions and feedback, enabling e-commerce businesses to "
    "monitor and enhance customer satisfaction in real-time."
)
st.sidebar.subheader("Model Details")
st.sidebar.write("- **Model:** Bidirectional LSTM")
st.sidebar.write("- **Input:** Customer Remarks (Text)")
st.sidebar.write("- **Output:** Predicted CSAT Score (1-5)")