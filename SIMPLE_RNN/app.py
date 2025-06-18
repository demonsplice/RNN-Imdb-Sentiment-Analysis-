# step1 import all the libraries 
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the word index from imdb dataset 
word_index = imdb.get_word_index()
# Reverse the word index to get words from indices  
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Set the maxlen used during model training
maxlen = 500  # Change this if you used a different value during training

# Load the model 
model = load_model("imdb_rnn_model.h5")

# Helper function to preprocess the user input review 
def preprocess_test(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Convert words to indices
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)  # Pad the sequence
    return padded_review

# Function to predict sentiment of a review
def predict_sentiment(review):
    # Preprocess the review
    preprocessed_review = preprocess_test(review)
    # Predict sentiment
    prediction = model.predict(preprocessed_review)
    sentiment_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment_label, prediction[0][0]

# Design a Streamlit app
st.title("IMDB Movie Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):")

user_input = st.text_area("Movie Review", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        # Predict sentiment directly from user input
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("Please enter a review to get the sentiment prediction.")