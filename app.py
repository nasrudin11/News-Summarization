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

    # Tambahkan langkah untuk menghapus 'kompas.com' dari teks sebelum tokenisasi
    text_cleaned = re.sub(r'kompas.com', '', text_cleaned)

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
def create_graph(cosine_sim):
    threshold = 0.2
    adjacency = (cosine_sim > threshold).astype(int)

    # Buat graf baru
    G = nx.Graph()

    # Tambahkan node (kalimat)
    for i in range(len(adjacency)):
        G.add_node(i)

    # Tambahkan edge (hubungan) antara kalimat berdasarkan matriks adjensi
    for i in range(len(adjacency)):
        for j in range(len(adjacency)):
            similarity = adjacency[i][j]  # Ambil nilai dari matriks adjacency
            if similarity > 0 and i != j:
                G.add_edge(i, j, weight=similarity)

    # Visualisasikan grafik dengan tata letak "circular"
    pos = nx.circular_layout(G)
    labels = {i: f'K {i}' for i in G.nodes()}

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=900, node_color='skyblue', edge_color='gray')
    plt.title('Graf Hubungan Kalimat Berdasarkan Matriks Adjacency')
    st.pyplot(plt)

    closeness_centrality = nx.closeness_centrality(G)
    closeness_centrality_df = pd.DataFrame(list(closeness_centrality.items()), columns=['Node', 'Closeness Centrality'])
    closeness_centrality_df = closeness_centrality_df.sort_values(by='Closeness Centrality', ascending=False)
    return closeness_centrality_df

# Streamlit app
st.title("Text Summarization Berbasis Graf dan Closeness Centrality")

# Input teks dari form
input_text = st.text_area("Masukkan Teks yang Ingin Dianalisis", height=200)

if st.button("Proses Teks"):
    if input_text:
        # Proses teks
        berita_no_stopwords, berita_tokenized = process_text(input_text)

        # Hitung cosine similarity
        vect_matrix, cosine_sim = compute_tfidf_and_similarity(berita_no_stopwords)

        # Tampilkan graf hubungan antar kalimat
        closeness_centrality_df = create_graph(cosine_sim)

        # Mengambil kalimat berdasarkan Closeness Centrality tertinggi
        top_three_closeness = closeness_centrality_df.head(3)
        top_node_closeness = top_three_closeness['Node'].apply(lambda x: int(x))

        # Gabungkan kalimat yang terpilih menjadi paragraf
        summary = ' '.join([berita_tokenized[node_index] for node_index in top_node_closeness if 0 <= node_index < len(berita_tokenized)])

        # Tampilkan hasil ringkasan dalam bentuk paragraf
        st.write("Ringkasan:")
        st.write(summary)
    else:
        st.warning("Masukkan teks terlebih dahulu!")
