import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
import nltk
import requests
from bs4 import BeautifulSoup

# Interface Streamlit
st.set_page_config(layout="wide")

st.title("Analyse d'entités nommées")


# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt', quiet=True)

# Chargement du modèle CRF
@st.cache_resource
def load_model():
    return joblib.load('best_crf_ner_model.joblib')

model = load_model()

# Fonction pour extraire les caractéristiques d'une phrase
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Fonction pour extraire les entités nommées
def extract_named_entities(text, crf_model):
    words = word_tokenize(text)
    features = sent2features([(word, '') for word in words])
    labels = crf_model.predict([features])[0]
    entities = []
    current_entity = []
    current_label = None
    for word, label in zip(words, labels):
        if label != 'O':
            if label != current_label:
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = [word]
                current_label = label
            else:
                current_entity.append(word)
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_label))
                current_entity = []
                current_label = None
    if current_entity:
        entities.append((' '.join(current_entity), current_label))
    return entities

def scrape_saabal():
    url = "https://saabal.net/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    texts = [article.get_text() for article in articles]
    return texts


tab1, tab2 = st.tabs(["Texte manuel", "Articles Saabal"])

with tab1:
    user_input = st.text_area("Entrez votre texte ici :")
    if st.button("Analyser", key="analyze_manual"):
        if user_input:
            entities = extract_named_entities(user_input, model)
            st.write("Entités extraites :")
            for entity, label in entities:
                st.write(f"{label}: {entity}")
        else:
            st.warning("Veuillez entrer du texte à analyser.")

with tab2:
    if st.button("Charger et analyser des articles", key="analyze_saabal"):
        with st.spinner("Chargement des articles..."):
            texts = scrape_saabal()
        for i, text in enumerate(texts, 1):
            with st.expander(f"Article {i}"):
                st.write(text[:200] + "...")
                entities = extract_named_entities(text, model)
                st.write("Entités extraites :")
                for entity, label in entities:
                    st.write(f"{label}: {entity}")

st.sidebar.info("Cette application utilise un modèle CRF pour la reconnaissance d'entités nommées dans des textes manuels ou des articles de Saabal.net.")
