import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Parking AI", layout="wide")

st.title("🅿️ Parking AI – Belegungsanalyse")

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)

st.sidebar.info("Erkennt: Autos im Bild und berechnet Belegung")

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
file = st.file_uploader("📤 Parkplatzbild hochladen", type=["jpg","png","jpeg"])

# ---------------------------------------------------
# FIXED PARKING SLOTS (WICHTIG!)
# ---------------------------------------------------
# Beispiel: 6 Parkplätze als Rechtecke
# (x1, y1, x2, y2) – MUSS ggf. an dein Bild angepasst werden
PARKING_SLOTS = [
    (50, 200, 200, 400),
    (220, 200, 370, 400),
    (390, 200, 540, 400),
    (560, 200, 710, 400),
    (730, 200, 880, 400),
    (900, 200, 1050, 400),
]

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if file:

    image = Image.open(file).convert("RGB")
    img = np.array(image)

    results = model.predict(img, conf=conf)[0]

    boxes = results.boxes

    occupied = [False] * len(PARKING_SLOTS)

    # ---------------------------------------------------
    # CHECK CARS
    # ---------------------------------------------------
    for box in boxes:

        cls = int(box.cls[0])
        name = model.names[cls]

        if name != "car":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        car_center = ((x1 + x2)//2, (y1 + y2)//2)

        # check in which slot car is
        for i, (sx1, sy1, sx2, sy2) in enumerate(PARKING_SLOTS):

            if sx1 < car_center[0] < sx2 and sy1 < car_center[1] < sy2:
                occupied[i] = True

    # ---------------------------------------------------
    # DRAW RESULT IMAGE
    # ---------------------------------------------------
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(PARKING_SLOTS):

        color = (0, 0, 255) if occupied[i] else (0, 255, 0)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

        label = "BELEGT" if occupied[i] else "FREI"

        cv2.putText(
            img_cv,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ---------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        st.image(img_cv, caption="Parking Analyse")

    # ---------------------------------------------------
    # STATS
    # ---------------------------------------------------
    total = len(PARKING_SLOTS)
    used = sum(occupied)
    free = total - used

    st.markdown("## 📊 Parkplatz Status")

    st.metric("Gesamt", total)
    st.metric("Belegt", used)
    st.metric("Frei", free)
    st.metric("Auslastung", f"{used/total*100:.1f}%")
