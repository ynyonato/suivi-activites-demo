
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
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from keybert import KeyBERT

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

st.set_page_config(page_title="Analyse Feedbacks + Clustering IA", layout="wide")
st.title("🧠 Analyse Automatisée de Feedbacks avec Thèmes IA")

uploaded_file = st.file_uploader("📤 Importez votre fichier CSV `suivi_activites.csv`", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Fichier chargé avec succès.")
    st.dataframe(df.head())

    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join([mot for mot in text.split() if mot not in stop_words])
        return text

    df['Feedback_clean'] = df['feedback'].apply(clean_text)

    # Sentiment
    def compute_sentiment(text):
        return TextBlob(text).sentiment.polarity

    df['sentiment'] = df['Feedback_clean'].apply(compute_sentiment)

    def classify_sentiment(score):
        if score > 0.1:
            return 'positif'
        elif score < -0.1:
            return 'négatif'
        else:
            return 'neutre'

    df['sentiment_cat'] = df['sentiment'].apply(classify_sentiment)

    # Wordcloud
    st.subheader("☁️ Nuage de mots")
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(' '.join(df['Feedback_clean']))
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # TF-IDF + Clustering
    st.subheader("📌 Clustering thématique + renommage automatique avec IA")
    vectorizer = TfidfVectorizer(max_features=500, max_df=0.8, min_df=5)
    X = vectorizer.fit_transform(df['Feedback_clean'])

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Renommage auto avec KeyBERT
    kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')
    cluster_labels_ai = {}
    for i in range(n_clusters):
        cluster_text = " ".join(df[df['Cluster'] == i]['Feedback_clean'])
        if cluster_text.strip():
            keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 3),
                                                 stop_words='french', top_n=1)
            if keywords:
                label = keywords[0][0].capitalize()
            else:
                label = f"Thème {i}"
        else:
            label = f"Thème {i}"
        cluster_labels_ai[i] = label
        st.markdown(f"**Cluster {i} ➜ {label}**")

    df['Cluster_Label'] = df['Cluster'].map(cluster_labels_ai)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X.toarray())
    df['Dim1'] = X_embedded[:, 0]
    df['Dim2'] = X_embedded[:, 1]

    fig = px.scatter(
        df, x='Dim1', y='Dim2',
        color='Cluster_Label',
        hover_data=['feedback', 'type_activite', 'région'],
        title="📍 Visualisation t-SNE avec thèmes IA"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Affichage enrichi
    st.subheader("📄 Données enrichies")
    st.dataframe(df[['id_activite', 'date', 'type_activite', 'région', 'feedback', 'Cluster_Label', 'sentiment_cat']])

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
