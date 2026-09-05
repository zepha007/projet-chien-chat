import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="Chien ou Chat ?", page_icon="", layout="centered")

st.title("Classificateur de Chiens et Chats ")
st.write("Importe une photo ci-dessous pour découvrir si notre intelligence artificielle y voit un chien ou un chat !")

# Lien direct vers ton fichier .h5 hébergé (sans espace)
MODEL_URL = "https://huggingface.co/zepha007/chien-chat-classifer/resolve/main/cats_vs_dogs_efficientnet_gray%20(1).h5"
MODEL_PATH = "modele_chiens_chats.h5"

@st.cache_resource
def load_my_model():
    # Télécharger le modèle s'il n'est pas déjà présent sur le serveur Streamlit
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle d'IA en cours (veuillez patienter)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Chargement du modèle
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Erreur lors du téléchargement ou du chargement du modèle : {e}")
    model = None

uploaded_file = st.file_uploader("Choisis une image (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image sélectionnée', use_container_width=True)
    
    if st.button('Lancer la prédiction'):
        with st.spinner("Analyse de l'image..."):
            image = image.resize((150, 150))
            img_array = np.array(image) / 255.0
            
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
                
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            if score > 0.5:
                st.success(f"C'est un Chat !  (Confiance : {score*100:.2f}%)")
            else:
                st.success(f"C'est un Chien !  (Confiance : {(1-score)*100:.2f}%)")
