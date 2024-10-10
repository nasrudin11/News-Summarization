import numpy as np
import pandas as pd
import nltk
import re
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk memproses teks
def process_text(text):
    # Ubah teks menjadi lowercase
    text_lower = text.lower()

    # Bersihkan teks dengan menghapus tanda baca dan karakter khusus
    text_cleaned = re.sub(r'[^\w\s.]', '', text_lower)
    text_cleaned = text_cleaned.replace('\n', ' ')

    # Tokenisasi kalimat
    sentences = sent_tokenize(text_cleaned)

    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = {'kompas.com'}
    stop_words.update(custom_stopwords)

    sentences_no_stopwords = []
    for sentence in sentences:
        words = sentence.split()
        words_no_stop = [word for word in words if word not in stop_words]
        cleaned_sentence = ' '.join(words_no_stop)
        sentences_no_stopwords.append(cleaned_sentence)

    return sentences_no_stopwords, sentences

# Fungsi untuk menghitung TF-IDF dan cosine similarity
def compute_tfidf_and_similarity(sentences_no_stopwords):
    vect = TfidfVectorizer()
    vect_matrix = vect.fit_transform(sentences_no_stopwords)
    cosine = cosine_similarity(vect_matrix, vect_matrix)
    return vect_matrix, cosine

# Fungsi untuk membuat adjacency matrix dan graf
def create_graph(cosine_sim, sentences):
    threshold = 0.2
    adjacency = (cosine_sim > threshold).astype(int)
    
    # Gunakan `from_numpy_array` untuk versi terbaru networkx
    G_adj = nx.from_numpy_array(adjacency)

    plt.figure(figsize=(8, 6))
    pos = nx.shell_layout(G_adj)
    nx.draw(G_adj, pos, with_labels=True, edge_color='black', node_color="skyblue", node_size=1000)
    plt.title('Graf Hubungan Kalimat Berdasarkan Matriks Adjacency')
    st.pyplot(plt)

    closeness_centrality = nx.closeness_centrality(G_adj)
    closeness_centrality_df = pd.DataFrame(list(closeness_centrality.items()), columns=['Node', 'Closeness Centrality'])
    closeness_centrality_df = closeness_centrality_df.sort_values(by='Closeness Centrality', ascending=False)
    return closeness_centrality_df

# Streamlit app
st.title("Analisis Teks Berbasis TF-IDF dan Graf")

# Input teks dari form
input_text = st.text_area("Masukkan Teks yang Ingin Dianalisis", height=200)

if st.button("Proses Teks"):
    if input_text:
        # Proses teks
        berita_no_stopwords, berita_tokenized = process_text(input_text)

        # Tampilkan kalimat yang telah dihilangkan stopwords-nya
        st.write("Kalimat Setelah Penghapusan Stopwords:")
        st.write(berita_no_stopwords)

        # Hitung cosine similarity
        vect_matrix, cosine_sim = compute_tfidf_and_similarity(berita_no_stopwords)

        # Tampilkan matriks cosine similarity
        st.write("Matriks Cosine Similarity:")
        st.write(cosine_sim)

        # Tampilkan graf hubungan antar kalimat
        st.write("Graf Hubungan Kalimat Berdasarkan Matriks Adjacency:")
        closeness_centrality_df = create_graph(cosine_sim, berita_tokenized)

        # Tampilkan Closeness Centrality
        st.write("Closeness Centrality:")
        st.write(closeness_centrality_df)

        # Mengambil kalimat berdasarkan Closeness Centrality tertinggi
        st.write("Kalimat yang Paling Menonjol Berdasarkan Closeness Centrality:")
        top_three_closeness = closeness_centrality_df.head(3)
        top_node_closeness = top_three_closeness['Node'].apply(lambda x: int(x))

        # Tampilkan kalimat yang sesuai dengan node-node teratas
        extracted_sentences = [berita_tokenized[node_index] for node_index in top_node_closeness if 0 <= node_index < len(berita_tokenized)]
        for idx, sentence in enumerate(extracted_sentences):
            st.write(f"K{idx}: {sentence}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")
