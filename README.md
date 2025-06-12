# 🧠 Analyse de Feedbacks avec Clustering et Génération de Thèmes par IA

Cette application Streamlit permet :
- D'importer un fichier CSV de suivi d'activités
- D'analyser automatiquement les feedbacks (nuage de mots, sentiments, clustering)
- De renommer chaque groupe (cluster) de feedbacks avec un **thème généré automatiquement par un modèle GPT (OpenAI)**

---

## 🚀 Fonctionnalités principales

- 📤 Téléversement d'un fichier `suivi_activites.csv`
- 🔍 Nettoyage automatique des feedbacks
- 🧠 Analyse de sentiment (positif / neutre / négatif)
- 📊 Visualisations (sentiment par activité, localisation, etc.)
- 🔀 Clustering avec TF-IDF + KMeans + t-SNE
- 🏷️ Thématisation automatique des clusters via LLM (GPT-3.5)
- 📈 Tendance mensuelle des sentiments
- 📄 Données enrichies affichables

---

## 🗂 Format attendu du CSV

Nom du fichier : `suivi_activites.csv`  
Colonnes attendues :

| id_activite | date       | type_activite       | nombre_participants | localisation | région   | feedback                  |
|-------------|------------|---------------------|----------------------|--------------|----------|---------------------------|
| 1           | 2025-01-10 | Atelier formation   | 18                   | Lomé         | Maritime | Très bonne organisation   |

---

## 🧪 Exécution locale

```bash
git clone https://github.com/votre-utilisateur/feedback-cluster-ia.git
cd feedback-cluster-ia

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
