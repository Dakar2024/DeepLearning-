import logging
import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Liste des noms de classes
class_names = ["Uninfected", "Parasitized"]
dog_cat_class_names = ["Chat", "Chien"]

# CSS pour une sidebar épurée
st.markdown("""
    <style>
    .sidebar-title {
        font-size: 28px;
        font-weight: bold;
        color: #4F8BF9;
        margin-bottom: 30px;
    }
    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .sidebar-btn {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 500;
        color: #e0e0e0;
        background: none;
        border: none;
        text-align: left;
        cursor: pointer;
        transition: background 0.2s, color 0.2s, font-weight 0.2s, box-shadow 0.2s;
        text-decoration: none;
        outline: none;
    }
    .sidebar-btn.active {
        background: rgba(255,255,255,0.15);
        color: #fff;
        font-weight: bold;
        box-shadow: 0 0 16px 2px rgba(255,255,255,0.5);
    }
    .sidebar-btn:hover {
        background: rgba(255,255,255,0.08);
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

pages = [
    ("Accueil", "🏠"),
    ("Chien/Chat", "🐶"),
    ("Paludisme", "🦠")
]

if "page" not in st.session_state:
    st.session_state.page = "Accueil"
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Afficher le logo au-dessus du menu
st.sidebar.image("logo.png", width=120)
st.sidebar.markdown('<div class="sidebar-title">Menu</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)

for name, emoji in pages:
    active = "active" if st.session_state.page == name else ""
    button_clicked = st.sidebar.button(f"{emoji} {name}", key=f"btn_{name}")
    st.markdown(f"""<style>
        [data-testid=\"stSidebar\"] button[data-testid=\"baseButton-secondary\"]#btn_{name} {{
            {'background: rgba(255,255,255,0.15); color: #fff; font-weight: bold; box-shadow: 0 0 16px 2px rgba(255,255,255,0.5);' if active else ''}
        }})
        </style>
    """, unsafe_allow_html=True)
    if button_clicked:
        st.session_state.page = name
st.sidebar.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page


def get_cells_model():
    if "cells_model" not in st.session_state:
        import tensorflow as tf
        st.session_state.cells_model = tf.keras.models.load_model(
            "./Modeles/cells_model.keras")
    return st.session_state.cells_model


def get_dog_cat_model():
    if "dog_cat_model" not in st.session_state:
        import tensorflow as tf
        st.session_state.dog_cat_model = tf.keras.models.load_model(
            "./Modeles/dog_cat_model.keras")
    return st.session_state.dog_cat_model


if page == "Accueil":
    st.title("Détection d'images par Deep Learning")
    st.markdown("""
**Description du projet de Deep Learning**

Dans le cadre de notre projet académique de Deep Learning, nous avons conçu et entraîné deux modèles de classification d'images répondant à des problématiques réelles :

**Détection de cellules infectées par le paludisme**  
Ce modèle exploite un jeu de données d'images microscopiques pour distinguer les cellules infectées des cellules saines, en vue de faciliter l'aide au diagnostic médical automatisé.

**Classification d'images de chiens et de chats**  
Ce deuxième modèle a été entraîné pour identifier si une image représente un chien ou un chat, à partir d'un ensemble d'images annotées.

Pour rendre ces modèles accessibles à des utilisateurs non techniques, nous avons développé une application web avec Streamlit. Elle permet :
- L'upload d'images via une interface intuitive,
- L'exécution des prédictions en temps réel,


Le projet s'inscrit dans un cadre académique rigoureux avec :
- Une deadline prolongée jusqu'au 05 juillet 2025,
- L'exigence de constituer des groupes de deux maximum,
- La production d'un notebok détaillant les étapes d'implémentation ,
- Presentation powerpoint en ligne ,avec deux notes distinctes (implémentation & présentation), 


Ce projet nous a permis de mettre en œuvre des compétences pratiques en Deep Learning, en développement d'application interactive.
    """)

elif page == "Chien/Chat":
    st.title("Détection Chien/Chat")
    uploaded_files = st.file_uploader("Uploader une ou plusieurs images de chien ou chat", type=[
                                      "jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.write("Résultats de la détection :")
        dog_cat_model = get_dog_cat_model()
        for uploaded_file in uploaded_files:
            image_obj = Image.open(uploaded_file)
            st.session_state.uploaded_images.append({
                "image": image_obj,
                "category": "Chien/Chat",
                "date": datetime.now().date()
            })
            st.image(image_obj, caption="Image uploadée",
                     use_container_width=True)

            img = image_obj.convert("RGB").resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = dog_cat_model.predict(img_array)
            if prediction.shape[1] == 1:
                proba = prediction[0][0]
                predicted_label = dog_cat_class_names[int(proba > 0.5)]
                confidence = proba if proba > 0.5 else 1 - proba
                if confidence < 0.6:
                    st.error(
                        "❌ Veuillez uploader une image correspondant au modèle (chien ou chat). Confiance trop faible.")
                else:
                    st.success(
                        f"🐾 Prédiction : **{predicted_label}** — Confiance : {proba:.2%}")
            else:
                predicted_index = int(np.argmax(prediction))
                confidence = prediction[0][predicted_index]
                predicted_label = dog_cat_class_names[predicted_index]
                if confidence < 0.6:
                    st.error(
                        "❌ Veuillez uploader une image correspondant au modèle (chien ou chat). Confiance trop faible.")
                else:
                    st.success(
                        f"🐾 Prédiction : **{predicted_label}** — Confiance : {confidence:.2%}")

elif page == "Paludisme":
    st.title("Détection Paludisme")
    uploaded_files = st.file_uploader(
        "Uploader une ou plusieurs images de cellules de paludisme",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.write("Résultats de la détection :")
        cells_model = get_cells_model()
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_images.append({
                "image": image,
                "category": "Paludisme",
                "date": datetime.now().date()
            })
            st.image(image, caption="Image uploadée", use_container_width=True)

            img = image.convert("RGB").resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cells_model.predict(img_array)
            if prediction.shape[1] == 1:
                proba = prediction[0][0]
                if proba > 0.5:
                    predicted_label = class_names[1]
                    confidence = proba
                else:
                    predicted_label = class_names[0]
                    confidence = 1 - proba
                if confidence < 0.6:
                    st.error(
                        "❌ Veuillez uploader une image correspondant au modèle (cellule de paludisme). Confiance trop faible.")
                else:
                    st.success(
                        f"🧪 Prédiction : **{predicted_label}** — Confiance : {confidence:.2%}")
            else:
                predicted_index = int(np.argmax(prediction))
                confidence = float(prediction[0][predicted_index])
                predicted_label = class_names[predicted_index]
                if confidence < 0.6:
                    st.error(
                        "❌ Veuillez uploader une image correspondant au modèle (cellule de paludisme). Confiance trop faible.")
                else:
                    st.success(
                        f"🧪 Prédiction : **{predicted_label}** — Confiance : {confidence:.2%}")
