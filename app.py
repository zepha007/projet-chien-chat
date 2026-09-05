import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="Chien ou Chat ?", page_icon="", layout="centered")

st.title("Classificateur de Chiens et Chats ")
st.write("Importe une photo ci-dessous pour découvrir si notre intelligence artificielle y voit un chien ou un chat !")

# Lien direct vers ton modèle en niveaux de gris sur Hugging Face
MODEL_URL = "https://huggingface.co/zepha007/chien-chat-classifer/resolve/main/cats_vs_dogs_efficientnet_gray%20(1).h5"
MODEL_PATH = "modele_chiens_chats.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle d'IA en cours (veuillez patienter)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
            # Traitement strictement identique à ton script local (Niveaux de gris 'L')
            img = image.convert('L').resize((150, 150))
            tab_img = np.array(img) / 255.0
            donnees_finales = tab_img.reshape(1, 150, 150, 1)
            
            prediction = model.predict(donnees_finales, verbose=0)
            
            # Logique exacte de ton script local pour chien/chat et score
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
            
            st.success(f"C'est un {animal} !  (Confiance : {certitude*100:.2f}%)")
