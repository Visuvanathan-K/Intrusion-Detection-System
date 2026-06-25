import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from tensorflow.keras.models import load_model

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Intrusion Detection System using LSTM")

st.write(
    "Upload a network traffic dataset and classify each record as "
    "Non-Tor, NonVPN, Tor, or VPN."
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_resources():

    model = load_model("models/intrusion_detection_lstm.keras")

    scaler = joblib.load("models/scaler.pkl")

    label_encoder = joblib.load("models/label_encoder.pkl")

    return model, scaler, label_encoder


model, scaler, label_encoder = load_resources()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls", "txt", "tsv"]
)

if uploaded_file is not None:

    extension = os.path.splitext(uploaded_file.name)[1].lower()

    try:

        if extension == ".csv":

            df = pd.read_csv(uploaded_file)

        elif extension in [".xlsx", ".xls"]:

            df = pd.read_excel(uploaded_file)

        elif extension == ".tsv":

            df = pd.read_csv(uploaded_file, sep="\t")

        elif extension == ".txt":

            try:

                df = pd.read_csv(uploaded_file)

            except:

                uploaded_file.seek(0)

                df = pd.read_csv(uploaded_file, sep="\t")

        else:

            st.error("Unsupported File Format")

            st.stop()

    except Exception as e:

        st.error(e)

        st.stop()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ----------------------------------------
    # Remove unnecessary columns
    # ----------------------------------------

    columns_to_drop = [
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Timestamp",
        "Label",
        "Label.1"
    ]

    df.drop(
        columns=columns_to_drop,
        errors="ignore",
        inplace=True
    )

    # ----------------------------------------
    # Missing Values
    # ----------------------------------------

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    if st.button("Predict"):

        try:

            X = scaler.transform(df)

            X = X.reshape(
                X.shape[0],
                1,
                X.shape[1]
            )

            prediction = model.predict(X)

            prediction = np.argmax(
                prediction,
                axis=1
            )

            prediction = label_encoder.inverse_transform(
                prediction
            )

            result = df.copy()

            result["Prediction"] = prediction

            st.success("Prediction Completed")

            st.subheader("Prediction Results")

            st.dataframe(result)

            st.subheader("Prediction Count")

            st.bar_chart(
                result["Prediction"].value_counts()
            )

            csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Prediction",
                csv,
                "prediction.csv",
                "text/csv"
            )

        except Exception as e:

            st.error(e)