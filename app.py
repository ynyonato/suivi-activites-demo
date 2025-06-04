
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
import string
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.corpus import stopwords

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

st.set_page_config(page_title="Analyse de Feedbacks - Suivi d'Activités", layout="wide")
st.title("📊 Analyse Automatisée de Feedbacks (Suivi & Évaluation de Projets)")

# Upload de fichier CSV
uploaded_file = st.file_uploader("📤 Importez votre fichier CSV `suivi_activites.csv`", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Fichier chargé avec succès.")
    st.subheader("👁️ Aperçu des données brutes")
    st.dataframe(df.head())

    # Nettoyage du texte
    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join([mot for mot in text.split() if mot not in stop_words])
        return text

    df['Feedback_clean'] = df['feedback'].apply(clean_text)

    # Wordcloud
    st.subheader("☁️ Nuage de mots sur les feedbacks")
    text = ' '.join(df['Feedback_clean'])
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # TF-IDF + clustering
    st.subheader("📌 Clustering thématique avec TF-IDF + KMeans + t-SNE")
    vectorizer = TfidfVectorizer(max_features=500, max_df=0.8, min_df=5)
    X = vectorizer.fit_transform(df['Feedback_clean'])

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X.toarray())
    df['Dim1'] = X_embedded[:, 0]
    df['Dim2'] = X_embedded[:, 1]

    fig = px.scatter(df, x='Dim1', y='Dim2', color=df['Cluster'].astype(str),
                     hover_data=['feedback', 'région', 'type_activite'],
                     title="📍 Visualisation t-SNE des clusters de feedbacks")
    st.plotly_chart(fig, use_container_width=True)

    # Feedbacks par thème
    st.subheader("📊 Nombre de feedbacks par thème (Cluster)")
    st.bar_chart(df['Cluster'].value_counts().sort_index())

    # Analyse par région
    st.subheader("🗺️ Activités par région")
    region_counts = df['région'].value_counts()
    fig2 = px.bar(region_counts, x=region_counts.index, y=region_counts.values,
                  labels={'x': 'Région', 'y': "Nombre d'activités"},
                  title="Nombre d'activités par région")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📄 Données enrichies (Feedbacks + Clusters)")
    st.dataframe(df[['id_activite', 'date', 'type_activite', 'région', 'feedback', 'Cluster']])

else:
    st.info("Veuillez importer un fichier CSV pour démarrer l’analyse.")
