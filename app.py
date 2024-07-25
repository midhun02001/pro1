# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the vectorizer, model, and label encoder
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except ModuleNotFoundError as e:
    st.error(f"Missing module: {e}. Please install the required module using 'pip install scikit-learn'.")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Streamlit app
st.title('Story Genre Prediction')
st.write('Enter the summary of the story to predict its genre')

# Input from user
summary_input = st.text_area('Story Summary')

if st.button('Predict Genre'):
    if summary_input:
        try:
            if 'vectorizer' in globals() and 'model' in globals() and 'label_encoder' in globals():
                # Vectorize the input
                summary_tfidf = vectorizer.transform([summary_input])
                
                # Predict the genre
                genre_encoded = model.predict(summary_tfidf)
                genre = label_encoder.inverse_transform(genre_encoded)
                
                st.write(f'The predicted genre is: {genre[0]}')
            else:
                st.error("Model components not loaded correctly. Please check the pickle files.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.write('Please enter a summary to get a prediction')
