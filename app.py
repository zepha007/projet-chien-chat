import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
from datetime import datetime

st.set_page_config(page_title="Neural Vision Engine", page_icon=None, layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #030712;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    header, footer {visibility: hidden;}
    
    .block-container {
        padding: 3rem 2rem;
        max-width: 1400px;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.7) 0%, rgba(3, 7, 18, 0.7) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
    }

    h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -0.03em;
        background: linear-gradient(to right, #ffffff, #9ca3af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h3 {
        color: #f9fafb;
        font-weight: 600;
        font-size: 1.15rem;
        letter-spacing: -0.01em;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.7rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        border-color: rgba(255, 255, 255, 0.2);
    }

    div[data-testid="stFileUploader"] {
        background-color: rgba(17, 24, 39, 0.5);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
    }

    .stAlert {
        background-color: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f3f4f6 !important;
        border-radius: 10px;
        backdrop-filter: blur(8px);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Neural Vision Engine")
st.write("Plateforme d'analyse prédictive et de classification d'images par réseau de neurones profond.")
st.write("")

MODEL_URL = "https://huggingface.co/zepha007/chien-chat-classifer/resolve/main/cats_vs_dogs_efficientnet_gray%20(1).h5"
MODEL_PATH = "modele_chiens_chats.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement des poids du modèle..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Erreur critique lors du chargement du modèle : {e}")
    model = None

if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("Flux d'entrée (Source)")
        uploaded_file = st.file_uploader("Glissez ou sélectionnez une image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Aperçu de l\'échantillon', use_container_width=True)
        
        predict_btn = st.button("Lancer l'inférence neuronale")

with col2:
    with st.container(border=True):
        st.subheader("Console de résultats et logs")
        
        if predict_btn and uploaded_file is not None and model is not None:
            with st.spinner("Exécution du modèle en cours..."):
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
            st.success(f"Détection validée : **{latest['animal']}** (Précision : `{latest['confiance']}`)")
            
            st.write("")
            st.markdown("##### Historique des requêtes")
            
            for item in st.session_state.history:
                st.markdown(f"<span style='color: #9ca3af;'>[{item['date']}]</span> &nbsp; **{item['animal']}** &nbsp;|&nbsp; `Confiance: {item['confiance']}`", unsafe_allow_html=True)
            
            st.write("")
            if st.button("Purger l'historique"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("En attente d'un flux d'image pour initialiser l'analyse.")
            if st.button("Purger l'historique", disabled=True):
                pass
