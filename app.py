import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="SolarGuard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LIGHTWEIGHT APPEARANCE
# =========================================================

st.markdown(
    """
    <style>
    
    .stApp {
        background-color: #f5f8fb;
    }

    [data-testid="stSidebar"] {
        background-color: #eef3f7;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #0b3954;
        margin-bottom: 0px;
    }

    .main-subtitle {
        font-size: 18px;
        color: #526675;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("☀️ SolarGuard")

st.subheader(
    "Solar Panel Fault Detection using Deep Learning"
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "best_solar_panel_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("☀️ SolarGuard")

    st.write(
        "Solar panel condition classification "
        "using deep learning."
    )

    st.divider()

    st.subheader("🔍 Detection Classes")

    st.write("🐦 Bird-drop")
    st.write("✅ Clean")
    st.write("🌫️ Dusty")
    st.write("⚡ Electrical-damage")
    st.write("🔨 Physical-Damage")
    st.write("❄️ Snow-Covered")

    st.divider()

    st.subheader("🧠 Model Information")

    st.write("Architecture: **MobileNetV2**")
    st.write("Transfer Learning: **Yes**")
    st.write("Input Size: **224 × 224**")
    st.write("Number of Classes: **6**")

    st.divider()

    st.caption(
        "SolarGuard • Machine Learning Mini Project"
    )


# =========================================================
# UPLOAD
# =========================================================

st.header("📤 Upload Solar Panel Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.divider()


    # -----------------------------------------------------
    # IMAGE + RESULT
    # -----------------------------------------------------

    image_col, result_col = st.columns(
        [1, 1]
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with image_col:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    resized_image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # IMPORTANT:
    # The trained model already contains
    # Rescaling(1/127.5, offset=-1).
    #
    # Therefore, DO NOT normalize the image here.


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )


    predicted_index = int(
        np.argmax(predictions[0])
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = float(
        predictions[0][predicted_index] * 100
    )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    with result_col:

        st.subheader("🎯 Detection Result")

        st.success(
            f"Detected Condition: {predicted_class}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        if confidence >= 80:

            st.success(
                "🟢 High-confidence prediction"
            )

        elif confidence >= 50:

            st.warning(
                "🟡 Medium-confidence prediction"
            )

        else:

            st.error(
                "🔴 Low-confidence prediction. "
                "Try a clearer image."
            )


    # =====================================================
    # PROBABILITY ANALYSIS
    # =====================================================

    st.divider()

    st.header("📊 Prediction Probability")


    probability_data = []


    for i, class_name in enumerate(
        class_names
    ):

        probability = float(
            predictions[0][i]
        )

        probability_data.append(
            {
                "Condition": class_name,
                "Probability (%)": round(
                    probability * 100,
                    2
                )
            }
        )


    probability_df = pd.DataFrame(
        probability_data
    )


    chart_col, details_col = st.columns(
        [1.3, 0.7]
    )


    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    with chart_col:

        st.subheader(
            "Prediction Distribution"
        )

        chart_df = (
            probability_df
            .set_index("Condition")
        )

        st.bar_chart(
            chart_df,
            height=350
        )


    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    with details_col:

        st.subheader(
            "Class Probabilities"
        )


        for i, class_name in enumerate(
            class_names
        ):

            probability = float(
                predictions[0][i]
            )

            percentage = (
                probability * 100
            )


            st.write(
                f"**{class_name}**"
            )


            st.progress(
                probability
            )


            st.caption(
                f"{percentage:.2f}%"
            )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.divider()

    st.header("📋 Detection Summary")


    summary1, summary2, summary3 = st.columns(
        3
    )


    with summary1:

        st.metric(
            "Detected Condition",
            predicted_class
        )


    with summary2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    with summary3:

        st.metric(
            "Model",
            "MobileNetV2"
        )


# =========================================================
# NO IMAGE
# =========================================================

else:

    st.info(
        "☀️ Upload a solar panel image above "
        "to start detection."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "☀️ SolarGuard | "
    "Solar Panel Fault Detection using Deep Learning | "
    "Machine Learning Mini Project"
)