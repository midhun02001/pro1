# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the vectorizer, model, and label encoder
vectorizer = None
model = None
label_encoder = None

try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("File not found: vectorizer.pkl")
except Exception as e:
    st.error(f"An error occurred while loading vectorizer: {e}")

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("File not found: model.pkl")
except Exception as e:
    st.error(f"An error occurred while loading model: {e}")

try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("File not found: label_encoder.pkl")
except Exception as e:
    st.error(f"An error occurred while loading label_encoder: {e}")

if vectorizer and model and label_encoder:
    st.title('Story Genre Prediction')
    st.write('Enter the summary of the story to predict its genre')

    # Input from user
    summary_input = st.text_area('Story Summary')

    if st.button('Predict Genre'):
        if summary_input:
            try:
                # Vectorize the input
                summary_tfidf = vectorizer.transform([summary_input])
                
                # Predict the genre
                genre_encoded = model.predict(summary_tfidf)
                genre = label_encoder.inverse_transform(genre_encoded)
                
                st.write(f'The predicted genre is: {genre[0]}')
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.write('Please enter a summary to get a prediction')
else:
    st.error("Model components not loaded correctly. Please check the pickle files.")
