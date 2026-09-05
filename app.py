import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
from datetime import datetime

st.set_page_config(page_title="Classification Chiens et Chats", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
        font-size: 1.8rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Classification de Chiens et Chats")
st.write("Interface de traitement et d'analyse d'images.")
st.write("")

MODEL_URL = "https://huggingface.co/zepha007/chien-chat-classifer/resolve/main/cats_vs_dogs_efficientnet_gray%20(1).h5"
MODEL_PATH = "modele_chiens_chats.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Chargement du modele en cours..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modele : {e}")
    model = None

if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.container(border=True):
        st.subheader("Selection de l'image")
        uploaded_file = st.file_uploader("Choisir un fichier image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Aperçu', use_container_width=True)
        
        predict_btn = st.button("Lancer l'analyse")

with col2:
    with st.container(border=True):
        st.subheader("Resultats et historique")
        
        if predict_btn and uploaded_file is not None and model is not None:
            with st.spinner("Traitement en cours..."):
                img = image.convert('L').resize((150, 150))
                tab_img = np.array(img) / 255.0
                donnees_finales = tab_img.reshape(1, 150, 150, 1)
                
                prediction = model.predict(donnees_finales, verbose=0)
                
                if prediction.shape[1] == 1:
                    score_final = float(prediction[0][0])
                    if score_final > 0.5:
                        animal, certitude = "Chien", score_final
                    else:
                        animal, certitude = "Chat", 1 - score_final
                else:
                    choix = np.argmax(prediction[0])
                    noms_classes = ["Chat", "Chien"]
                    animal = noms_classes[choix]
                    certitude = float(prediction[0][choix])
                
                conf_percent = certitude * 100
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.history.insert(0, {
                    "animal": animal,
                    "confiance": f"{conf_percent:.2f}%",
                    "date": timestamp
                })

        if len(st.session_state.history) > 0:
            latest = st.session_state.history[0]
            st.success(f"Resultat : {latest['animal']} (Indice de confiance : {latest['confiance']})")
            
            st.write("")
            st.text("Historique des sessions")
            
            for item in st.session_state.history:
                st.write(f"- {item['animal']} | Confiance : {item['confiance']} | {item['date']}")
            
            st.write("")
            if st.button("Effacer l'historique"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Veuillez importer une image pour afficher les resultats.")
            if st.button("Effacer l'historique", disabled=True):
                pass
