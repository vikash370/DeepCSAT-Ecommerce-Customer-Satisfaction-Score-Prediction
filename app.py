import streamlit as st
import numpy as np
import json
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# --- NLTK Data Download ---
# This runs once to get the necessary NLTK data
@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
load_nltk_data()

# --- Load All Artifacts (Model, Tokenizer) ---
# st.cache_resource ensures this only runs once
@st.cache_resource
def load_app_artifacts():
    model = load_model('deep_csat_text_model.h5')
    
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    max_length = 50  # Must be the same as in training!
    
    return model, tokenizer, stop_words, lemmatizer, max_length

# Load everything
model, tokenizer, stop_words, lemmatizer, max_length = load_app_artifacts()

# --- Text Preprocessing Function ---
def clean_and_preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_text = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    return padded_text

# --- Streamlit App UI ---

st.title('DeepCSAT: Customer Satisfaction Predictor 🤖')
st.write("Enter a customer's feedback to predict their satisfaction score (1-5).")

# User input text area
user_input = st.text_area("Customer Remark:", height=100)

# Prediction button
if st.button('Predict Score'):
    if user_input:
        try:
            # 1. Process input
            processed_input = clean_and_preprocess_text(user_input)
            
            # 2. Make Prediction
            prediction = model.predict(processed_input)
            predicted_score = float(prediction[0][0])
            
            # 3. Display result
            st.header(f"Predicted CSAT Score: {predicted_score:.2f} / 5.0")
            
            # Add a conditional message
            if predicted_score >= 4.5:
                st.success("This customer is very satisfied! 😊")
            elif predicted_score >= 3.5:
                st.info("This customer is neutral or mildly satisfied. 🙂")
            elif predicted_score >= 2.5:
                st.warning("This customer is dissatisfied. 😟")
            else:
                st.error("This customer is very unhappy. 😡 Action may be required.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")