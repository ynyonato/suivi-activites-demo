import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
import string
import re
import json
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from openai import OpenAI

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))
# Initialiser le client avec la clé
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

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
        
    # LLM to generate theme labels
    def generate_theme_from_feedbacks(feedbacks):
        joined = "\n".join([f"- {f}" for f in feedbacks])
        prompt = (
            f"Voici des retours d’utilisateurs :\n{joined}\n\n"
            f"En tant que Data Scientist Quel est le thème commun à ces retours qu'on peut degager ? "
            f"Réponds juste par un nom de thème clair, humainement compréhensible (4 mots max)."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
       
    
    # Checking matching between csv and json
    def valider_correspondance_geojson(df, geojson, csv_col='région', geojson_prop='region'):
        # Extraire les noms uniques du CSV
        regions_csv = set(df[csv_col].dropna().unique())
    
        # Extraire les noms du GeoJSON
        regions_geojson = set([f["properties"].get(geojson_prop, "").strip() for f in geojson["features"]])
    
        # Trouver les correspondances manquantes
        dans_csv_pas_geojson = regions_csv - regions_geojson
        dans_geojson_pas_csv = regions_geojson - regions_csv
    
        st.subheader("🔍 Validation des noms de régions")
        st.markdown(f"**✔️ Noms communs** : {regions_csv & regions_geojson}")
    
        if dans_csv_pas_geojson:
            st.error(f"❌ Ces régions sont dans le CSV mais pas dans le GeoJSON : {dans_csv_pas_geojson}")
        if dans_geojson_pas_csv:
            st.warning(f"⚠️ Ces régions sont dans le GeoJSON mais absentes du CSV : {dans_geojson_pas_csv}")
    
        if not dans_csv_pas_geojson and not dans_geojson_pas_csv:
            st.success("✅ Toutes les régions correspondent parfaitement.")
        
    # Conversion des données de la colonne Date en type date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    ### ANALYSE EXPLORATOIRE DE BASE (EDA)
    
    df['Feedback_clean'] = df['feedback'].apply(clean_text)
    df['sentiment'] = df['Feedback_clean'].apply(compute_sentiment)
    df['sentiment_cat'] = df['sentiment'].apply(classify_sentiment)
    
    st.subheader("📊 Indicateurs de suivi d'activités")
    # 📊 Bloc 1 : indicateurs rapides (métriques)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre total d’activités", len(df))
    col2.metric("Régions couvertes", df['région'].nunique())
    col3.metric("Localisations couvertes", df['localisation'].nunique())
    col4.metric("Score de sentiment moyen", round(df['sentiment'].mean(), 2))
    
    st.subheader("📈 Evolution du projet")
    col1, col2 = st.columns(2)
    with col1:
        # 4. Activités par localisation
        activites_par_localisation = df['localisation'].value_counts()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=activites_par_localisation.values, y=activites_par_localisation.index, palette='coolwarm', ax=ax3)
        ax3.set_title("📍 Nombre total d’activités par localisation")
        st.pyplot(fig3)
    
    with col2:
        # 3. Activités par région
        activites_par_region = df['région'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=activites_par_region.values, y=activites_par_region.index, palette='coolwarm', ax=ax2)
        ax2.set_title("📍 Nombre total d’activités par région")
        st.pyplot(fig2)  
    
    st.subheader("📈 Evolution Temporelle des activités")
    col3, col4, col5 = st.columns(3)
    with col3:
        # 5. Évolution par jour et région
        df_region = df.dropna(subset=['date', 'région'])
        activites_jour_region = df_region.groupby(['date', 'région']).size().reset_index(name='count')
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=activites_jour_region, x='date', y='count', hue='région', ax=ax4)
        ax4.set_title("📈 Activités par jour et région")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Nombre d’activités")
        st.pyplot(fig4)
    
    with col4:
        # 6. Évolution par jour et type d’activités
        df_type = df.dropna(subset=['date', 'type_activite'])
        activites_jour_type = df_type.groupby(['date', 'type_activite']).size().reset_index(name='count')
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=activites_jour_type, x='date', y='count', hue='type_activite', ax=ax5)
        ax5.set_title("📈 Activités par jour et type d’activité")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Nombre d’activités")
        st.pyplot(fig5)
        
    with col5:
         # 2. Nombre d’activités par jour
         activites_par_jour = df.groupby('date').size()
         fig1, ax1 = plt.subplots(figsize=(10, 4))
         sns.lineplot(data=activites_par_jour, ax=ax1)
         ax1.set_title("📅 Évolution du nombre d’activités par jour")
         ax1.set_xlabel("Date")
         ax1.set_ylabel("Nombre d’activités")
         st.pyplot(fig1)
    
    
    st.subheader("🗺️ Cartographies des sentiments")

    # Assurer que les données sont au bon format
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    col6, col7 = st.columns(2)
    with col6:
        # 1. Sentiment par localisation
        sentiment_loc = df.groupby('localisation')['sentiment'].mean().reset_index().dropna()
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=sentiment_loc, x='sentiment', y='localisation', palette='coolwarm', ax=ax1)
        ax1.set_title("Sentiment moyen par localisation")
        ax1.set_xlabel("Score de sentiment")
        ax1.set_ylabel("Localisation")
        st.pyplot(fig1)
        
    with col7:
        # 2. Sentiment par région
        sentiment_reg = df.groupby('région')['sentiment'].mean().reset_index().dropna()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=sentiment_reg, x='sentiment', y='région', palette='coolwarm', ax=ax2)
        ax2.set_title("Sentiment moyen par région")
        ax2.set_xlabel("Score de sentiment")
        ax2.set_ylabel("Région")
        st.pyplot(fig2)
    
    col8, col9 = st.columns(2)
    with col8:
        # 3. Sentiment par type d’activité
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

        st.markdown(commentaire)
        
    with col9:
        # Graphique des sentiments par type d'activité
        plt.figure(figsize=(10,6))
        sns.countplot(data=df, x='type_activite', hue='sentiment_cat', palette='coolwarm')
        plt.title("Sentiments par Type d'Activité \n")
        plt.xticks(rotation=35)
        #plt.show()
        st.pyplot(plt)
        
    col10, col11 = st.columns(2)
    with col10:
        # Croisement nombre participants vs sentiment (boxplot)
        plt.figure(figsize=(8,5))
        sns.boxplot(data=df, x='sentiment_cat', y='nombre_participants', linewidth=.75, palette='coolwarm')
        plt.title("Répartition des participants par sentiment \n")
        plt.xlabel("Catégorie de sentiment")
        plt.ylabel("Nombre de participants")
        plt.show()
        st.pyplot(plt)
    
    with col11:
        # Commentaires des boîtes à moustaches
        st.markdown("")
        
    col12, col13 = st.columns(2)
    with col12:
        # Wordcloud
        wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(' '.join(df['Feedback_clean']))
        plt.figure(figsize=(12, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
    with col13:
        # Commentaires du nuage de mots
        st.markdown("☁️ Nuage de mots")

    
    # TF-IDF + Clustering
    st.subheader("📌 Clustering thématique + renommage automatique avec IA")
    vectorizer = TfidfVectorizer(max_features=500, max_df=0.8, min_df=5)
    X = vectorizer.fit_transform(df['Feedback_clean'])

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Génération des labels par LLM
    cluster_labels_llm = {}
    for i in range(n_clusters):
         feedbacks = df[df['Cluster'] == i]['feedback'].dropna().sample(min(10, len(df[df['Cluster'] == i]))).tolist()
         try:
             label = generate_theme_from_feedbacks(feedbacks)
         except Exception as e:
             label = f"Thème {i}"
             st.warning(f"Erreur dans le cluster {i} : {e}")
         cluster_labels_llm[i] = label
         st.markdown(f"**Cluster {i} ➜ {label}**")
         
         df['Cluster_Label'] = df['Cluster'].map(cluster_labels_llm)

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
    
    # Affichage de la carte du Togo
    sentiment_par_region = df.groupby('région')['sentiment'].mean().reset_index()
    # Charger le fichier GeoJSON
    with open("togo_Regions_level_1.geojson", "r", encoding="utf-8") as f:
        togo_geo = json.load(f)
    
   # 🔍 1. Nettoyage de la propriété GeoJSON (retirer " Region")
    for feature in togo_geo['features']:
        region_name = feature['properties'].get('shape1', '')
        feature['properties']['region_clean'] = region_name.replace(" Region", "").strip()
    
    # ✅ 2. Validation (optionnelle mais utile)
    valider_correspondance_geojson(df, togo_geo, csv_col='région', geojson_prop='region_clean')
    
    # ✅ 3. Préparation des données sentiment par région
    sentiment_region = df.groupby('région')['sentiment'].mean().reset_index()
    sentiment_region.columns = ['region', 'sentiment']  # IMPORTANT : correspond au champ GeoJSON "region_clean"
    
    # ✅ 4. Création de la carte
    fig = px.choropleth(
        sentiment_region,
        geojson=togo_geo,
        featureidkey="properties.region_clean",  # correspondance personnalisée
        locations='region',
        color='sentiment',
        color_continuous_scale="RdYlGn",
        range_color=(-0.1, 0.1),
        labels={'sentiment': 'Score de sentiment'},
        title="💬 Sentiment moyen par région du Togo"
    )
    
    # 🔍 Zoom automatique sur les formes géographiques
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Affichage enrichi
    st.subheader("📄 Données enrichies")
    st.dataframe(df[['id_activite', 'date', 'type_activite', 'région', 'feedback', 'Cluster_Label', 'sentiment_cat']])

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
