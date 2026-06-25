import os
import joblib
import matplotlib.pyplot as plt

from src.preprocess import preprocess_data
from src.model import build_model


def train_model():

    # Dataset Path
    file_path = "data/Darknet.CSV"

    # Preprocess Data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoder
    ) = preprocess_data(file_path)

    print("\nData Preprocessing Completed!")
    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    # Build Model
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=len(label_encoder.classes_)
    )

    print("\nModel Summary")
    model.summary()

    # Train Model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        verbose=1
    )

    # Create folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save Model
    model.save("models/intrusion_detection_lstm.keras")

    # Save Scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Save Label Encoder
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print("\nModel Saved Successfully!")

    # Accuracy Plot
    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("outputs/training_accuracy.png")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/training_loss.png")
    plt.close()

    print("\nTraining Graphs Saved!")

    return model