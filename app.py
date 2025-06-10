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
from transformers import pipeline
from tqdm import tqdm


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


    # Sentiment
    def compute_sentiment(text):
        return TextBlob(text).sentiment.polarity


    def classify_sentiment(score):
        if score > 0.1:
            return 'positif'
        elif score < -0.1:
            return 'négatif'
        else:
            return 'neutre'
            
    def classer_sentiment(score):
        if score > 0.1:
            return 'POS'
        elif score < -0.1:
            return 'NEG'
        else:
            return 'NEU' 

    # Conversion des données de la colonne Date en type date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    ### ANALYSE EXPLORATOIRE DE BASE (EDA)
    
    df['Feedback_clean'] = df['feedback'].apply(clean_text)
    df['sentiment'] = df['Feedback_clean'].apply(compute_sentiment)
    df['sentiment_cat'] = df['sentiment'].apply(classify_sentiment)
        
    # Graphique des sentiments par type d'activité
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='type_activite', hue='sentiment_cat')
    plt.title("Sentiments par Type d'Activité \n")
    plt.xticks(rotation=45)
    #plt.show()
    st.pyplot(plt)

    # Moyenne de sentiment par localisation
    sentiment_localisation = df.groupby('localisation')['sentiment'].mean().sort_values()
    plt.figure(figsize=(8,5))
    sentiment_localisation.plot(kind='bar', color='skyblue')
    plt.title("Moyenne du sentiment par localisation \n")
    plt.ylabel("Score moyen de sentiment \n")
    #plt.show()
    st.pyplot(plt)

    # Croisement nombre participants vs sentiment (boxplot)
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='sentiment_cat', y='nombre_participants')
    plt.title("Nombre de participants selon sentiment \n")
    plt.show()
    st.pyplot(plt)

    # Wordcloud
    st.subheader("☁️ Nuage de mots")
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(' '.join(df['Feedback_clean']))
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
    
    # Nettoyage des feedbacks
    df = df.dropna(subset=['feedback'])  # Retirer les lignes sans feedback
    feedbacks = df['feedback'].astype(str).tolist()
    
    df['sentiment_label'] = df['sentiment'].apply(classer_sentiment)

    # Créer une copie filtrée pour analyse
    df_filtered = df[df['sentiment_label'].isin(['POS', 'NEG'])].copy()

    # Créer colonne mois-année
    df_filtered['mois_annee'] = df_filtered['date'].dt.to_period('M').astype(str)

    # Agrégation des feedbacks par mois et sentiment
    sentiment_par_mois = df_filtered.groupby(['mois_annee', 'sentiment_label']).size().unstack(fill_value=0)

    # Convertir l'index en entier pour les régressions
    mois_index = np.arange(len(sentiment_par_mois)).reshape(-1, 1)

    sentiment_par_mois = sentiment_par_mois.astype(int)

    # Tracer le graphique avec lignes de tendance
    plt.figure(figsize=(12, 6))

    sentiment_par_mois.plot(
        kind='bar',
        color={'POS': '#66bb6a', 'NEG': '#ef5350'},
        edgecolor='black',
        width=0.75,
        ax=plt.gca()
    )
    st.pyplot(plt)

    # Ajout des courbes de tendance
    for sentiment in ['POS', 'NEG']:
        y = sentiment_par_mois[sentiment].values
        model = LinearRegression().fit(mois_index, y)
        trend = model.predict(mois_index)
        plt.plot(sentiment_par_mois.index, trend, linestyle='--', linewidth=2, label=f"Tendance {sentiment}")

    # Finalisation
    plt.title("Évolution des sentiments par mois avec tendance \n")
    plt.xlabel("Mois-Année")
    plt.ylabel("Nombre de feedbacks")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Légende")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    # === Génération d'un commentaire automatique ===
    last = sentiment_par_mois.iloc[-1]
    first = sentiment_par_mois.iloc[0]

    evolution_pos = last['POS'] - first['POS']
    evolution_neg = last['NEG'] - first['NEG']

    commentaire = "🔎 **Analyse automatique :**\n"

    if evolution_pos > 0:
        commentaire += f"- Les feedbacks **positifs ont augmenté** de {evolution_pos} entre {sentiment_par_mois.index[0]} et {sentiment_par_mois.index[-1]}.\n"
    elif evolution_pos < 0:
        commentaire += f"- Les feedbacks **positifs ont diminué** de {-evolution_pos} sur la même période.\n"
    else:
        commentaire += "- Les feedbacks **positifs sont restés stables**.\n"

    if evolution_neg > 0:
        commentaire += f"- Les feedbacks **négatifs ont augmenté** de {evolution_neg}.\n"
    elif evolution_neg < 0:
        commentaire += f"- Les feedbacks **négatifs ont diminué** de {-evolution_neg}.\n"
    else:
        commentaire += "- Les feedbacks **négatifs sont restés stables**.\n"

    print(commentaire)

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
