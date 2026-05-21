import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline
import pandas as pd
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Parking AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

.title {
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:#38bdf8;
}

.subtitle {
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
}

.metric {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    text-align:center;
}

.metric-number {
    font-size:40px;
    color:#38bdf8;
    font-weight:bold;
}

.metric-label {
    color:#cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="title">🚗 Parking AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">KI-gestützte Parkplatz-Erkennung mit Hugging Face</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------
@st.cache_resource
def load_model():

    detector = pipeline(
        "image-classification",
        model="FGArmy/Parking_AI"
    )

    return detector

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.header("⚙️ Einstellungen")

    st.success("✅ Modell geladen")

    st.markdown("---")

    st.info("""
    📌 Funktionen:
    
    - Parkplatz-Erkennung
    - KI-Bildanalyse
    - Smart-City-System
    - Echtzeit-Auswertung
    """)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Parkplatzbild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ORIGINAL
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🖼️ Originalbild")

        st.image(
            image,
            use_column_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # KI ANALYSE
    with st.spinner("🧠 KI analysiert Parkplätze..."):

        time.sleep(1)

        predictions = model(image)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🎯 KI-Erkennung")

        st.image(
            image,
            use_column_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # RESULTS
    # ---------------------------------------------------
    st.markdown("## 📊 Analyse")

    top_prediction = predictions[0]

    label = top_prediction["label"]

    score = top_prediction["score"]

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{label}</div>
            <div class="metric-label">Erkennung</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{score:.2f}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # TABLE
    # ---------------------------------------------------
    st.markdown("## 📋 Alle Vorhersagen")

    df = pd.DataFrame(predictions)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    "<center>🚗 Parking AI | Streamlit + Hugging Face</center>",
    unsafe_allow_html=True
)
