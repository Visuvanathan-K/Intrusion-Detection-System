import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

from tensorflow.keras.models import load_model

from src.preprocess import preprocess_data


def evaluate_model():

    # ----------------------------------
    # Dataset Path
    # ----------------------------------
    file_path = "data/Darknet.CSV"

    # ----------------------------------
    # Load Preprocessed Data
    # ----------------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoder
    ) = preprocess_data(file_path)

    # ----------------------------------
    # Load Saved Model
    # ----------------------------------
    model = load_model("models/intrusion_detection_lstm.keras")

    print("\nModel Loaded Successfully!")

    # ----------------------------------
    # Predict
    # ----------------------------------
    y_pred_prob = model.predict(X_test)

    y_pred = np.argmax(y_pred_prob, axis=1)

    # ----------------------------------
    # Accuracy
    # ----------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    print("\n===================================")
    print(f"Accuracy : {accuracy:.4f}")
    print("===================================\n")

    # ----------------------------------
    # Classification Report
    # ----------------------------------
    print("Classification Report\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    # ----------------------------------
    # Confusion Matrix
    # ----------------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    plt.figure(figsize=(8, 6))

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    os.makedirs("outputs", exist_ok=True)

    plt.savefig("outputs/confusion_matrix.png")

    plt.show()

    print("\nConfusion Matrix Saved!")

    return accuracy