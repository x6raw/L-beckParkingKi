import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Parking AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------
# STYLE (clean & modern)
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
    font-size: 18px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
}

.metric-number {
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}

.metric-label {
    color: #cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="title">🚗 Parking AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 Parkplatz- & Fahrzeugerkennung</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL LOAD (STABLE FIX)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    # wichtig: immer .pt verwenden
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Einstellungen")

    conf = st.slider("Confidence", 0.1, 1.0, 0.25, 0.05)

    st.markdown("---")
    st.info("Erkennt: Autos, Busse, LKWs, Motorräder")

# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Parkplatzbild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Prediction
    results = model.predict(
        source=img_array,
        conf=conf,
        save=False
    )

    result_img = results[0].plot()

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Original")
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Erkennung")
        st.image(result_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # STATS
    # ---------------------------------------------------
    boxes = results[0].boxes

    vehicles = []
    vehicle_names = ["car", "truck", "bus", "motorcycle"]

    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf_score = float(box.conf[0])

        if name in vehicle_names:
            vehicles.append({
                "Objekt": name,
                "Confidence": round(conf_score, 2)
            })

    st.markdown("## 📊 Analyse")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{len(vehicles)}</div>
            <div class="metric-label">Fahrzeuge erkannt</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-number">{conf:.2f}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # TABLE
    # ---------------------------------------------------
    st.markdown("## 📋 Details")

    if vehicles:
        df = pd.DataFrame(vehicles)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Keine Fahrzeuge erkannt.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("<center>🚗 Parking AI | YOLOv8 + Streamlit</center>", unsafe_allow_html=True)
