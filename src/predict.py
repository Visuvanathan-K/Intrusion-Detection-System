import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model


def predict_network(sample_csv):

    # -------------------------
    # Load trained model
    # -------------------------
    model = load_model("models/intrusion_detection_lstm.keras")

    scaler = joblib.load("models/scaler.pkl")

    label_encoder = joblib.load("models/label_encoder.pkl")

    # -------------------------
    # Load new data
    # -------------------------
    df = pd.read_csv(sample_csv)

    # -------------------------
    # Remove unnecessary columns
    # -------------------------
    columns_to_drop = [
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Timestamp",
        "Label",
        "Label.1"
    ]

    df.drop(columns=columns_to_drop,
            errors="ignore",
            inplace=True)

    # -------------------------
    # Handle missing values
    # -------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    # -------------------------
    # Scale data
    # -------------------------
    X = scaler.transform(df)

    # -------------------------
    # Reshape for LSTM
    # -------------------------
    X = X.reshape(
        X.shape[0],
        1,
        X.shape[1]
    )

    # -------------------------
    # Predict
    # -------------------------
    prediction = model.predict(X)

    prediction = np.argmax(prediction, axis=1)

    prediction = label_encoder.inverse_transform(prediction)

    print("\nPrediction Results\n")

    for i, label in enumerate(prediction):

        print(f"Row {i+1} : {label}")

    return prediction