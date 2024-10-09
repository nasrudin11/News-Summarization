import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize text
    tokens = sent_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Main function to run the Streamlit app
def main():
    st.title("News Text Summarization")
    st.write("Aplikasi pemrosesan teks dan Perangkuman menggunakan Graph dan Clossnes Centrality.")

    # Get user input
    text_input = st.text_input("Enter some news text:")
    if text_input:
        # Preprocess the text
        preprocessed_text = preprocess_text(text_input)

        # Vectorize the preprocessed text
        X_text = vectorizer.transform([preprocessed_text])


        # Display summary
        st.write("Predicted News: Real News")

# Run the app
if __name__ == '__main__':
    main()
