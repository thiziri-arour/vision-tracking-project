"""
app.py — Dashboard Streamlit
Interface utilisateur qui consomme l'API FastAPI (backend/main.py) pour
lancer détection / segmentation / tracking sur images ou vidéos, et
afficher les résultats + statistiques.
"""
import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Vision Dashboard", layout="wide")
st.title("🎯 Dashboard Détection & Tracking d'Objets")

page = st.sidebar.radio("Navigation", ["Nouvelle analyse", "Historique des sessions"])

# ---------------------------------------------------------------------------
# Page 1 : lancer une nouvelle analyse
# ---------------------------------------------------------------------------
if page == "Nouvelle analyse":
    st.sidebar.header("Paramètres")
    task = st.sidebar.selectbox("Tâche", ["Détection (image)", "Segmentation (image)", "Tracking (vidéo)"])
    model_name = st.sidebar.text_input(
        "Modèle",
        value="yolov8n.pt" if task != "Segmentation (image)" else "yolov8n-seg.pt",
    )
    conf = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.05)

    if task == "Tracking (vidéo)":
        frame_skip = st.sidebar.number_input("Traiter 1 frame sur N", min_value=1, value=1)
        uploaded_file = st.file_uploader("Uploader une vidéo", type=["mp4", "avi", "mov"])
    else:
        uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Fichier source", width=350) if task != "Tracking (vidéo)" else None

        if st.button("Lancer l'analyse"):
            with st.spinner("Traitement en cours..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                params = {"model_name": model_name, "conf": conf}

                if task == "Détection (image)":
                    resp = requests.post(f"{API_URL}/detect", files=files, params=params)
                elif task == "Segmentation (image)":
                    resp = requests.post(f"{API_URL}/segment", files=files, params=params)
                else:
                    params["frame_skip"] = frame_skip
                    resp = requests.post(f"{API_URL}/track-video", files=files, params=params)

            if resp.status_code != 200:
                st.error(f"Erreur API : {resp.text}")
            else:
                data = resp.json()
                st.success(f"Analyse terminée — session #{data['session_id']}")

                col1, col2 = st.columns(2)

                if task == "Tracking (vidéo)":
                    with col1:
                        st.subheader("Vidéo annotée")
                        video_path = data["annotated_video_path"]
                        media_resp = requests.get(f"{API_URL}/media/{os.path.basename(video_path)}")
                        st.video(media_resp.content)
                    with col2:
                        st.subheader("Statistiques")
                        stats = data["stats"]
                        st.metric("Détections totales", stats["total_detections"])
                        st.metric("Objets uniques suivis (track IDs)", stats["unique_tracks"])
                        if stats["by_class"]:
                            df = pd.DataFrame(stats["by_class"])
                            st.bar_chart(df.set_index("class_name"))
                else:
                    with col1:
                        st.subheader("Image annotée")
                        image_path = data["annotated_image_path"]
                        media_resp = requests.get(f"{API_URL}/media/{os.path.basename(image_path)}")
                        st.image(media_resp.content)
                    with col2:
                        st.subheader("Détections")
                        st.json(data["counts"])
                        if data["detections"]:
                            st.dataframe(pd.DataFrame(data["detections"]))

# ---------------------------------------------------------------------------
# Page 2 : historique des sessions passées
# ---------------------------------------------------------------------------
else:
    st.header("Historique des sessions")
    resp = requests.get(f"{API_URL}/sessions")
    if resp.status_code != 200:
        st.error("Impossible de récupérer les sessions.")
    else:
        sessions = resp.json()
        if not sessions:
            st.info("Aucune session enregistrée pour le moment.")
        else:
            df = pd.DataFrame(sessions)
            st.dataframe(df, use_container_width=True)

            session_id = st.selectbox("Voir le détail d'une session", df["id"])
            if session_id:
                stats_resp = requests.get(f"{API_URL}/sessions/{session_id}/stats")
                if stats_resp.status_code == 200:
                    stats = stats_resp.json()
                    st.subheader(f"Session #{session_id}")
                    c1, c2 = st.columns(2)
                    c1.metric("Détections totales", stats["total_detections"])
                    c2.metric("Objets uniques suivis", stats["unique_tracks"])
                    if stats["by_class"]:
                        st.bar_chart(pd.DataFrame(stats["by_class"]).set_index("class_name"))
