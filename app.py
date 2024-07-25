# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the vectorizer, model, and label encoder
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Streamlit app
st.title('Story Genre Prediction')
st.write('Enter the summary of the story to predict its genre')

# Input from user
summary_input = st.text_area('Story Summary')

if st.button('Predict Genre'):
    if summary_input:
        # Vectorize the input
        summary_tfidf = vectorizer.transform([summary_input])
        
        # Predict the genre
        genre_encoded = model.predict(summary_tfidf)
        genre = label_encoder.inverse_transform(genre_encoded)
        
        st.write(f'The predicted genre is: {genre[0]}')
    else:
        st.write('Please enter a summary to get a prediction')
