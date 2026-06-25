import os
import joblib
import matplotlib.pyplot as plt


def create_folders():
    """
    Create required project folders.
    """

    folders = [
        "models",
        "outputs"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def save_scaler(scaler):
    """
    Save MinMaxScaler.
    """

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )


def load_scaler():
    """
    Load MinMaxScaler.
    """

    return joblib.load(
        "models/scaler.pkl"
    )


def save_label_encoder(label_encoder):
    """
    Save LabelEncoder.
    """

    joblib.dump(
        label_encoder,
        "models/label_encoder.pkl"
    )


def load_label_encoder():
    """
    Load LabelEncoder.
    """

    return joblib.load(
        "models/label_encoder.pkl"
    )


def save_training_plot(history):
    """
    Save Accuracy and Loss Graphs.
    """

    # Accuracy

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["accuracy"], label="Training Accuracy")

    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig("outputs/training_accuracy.png")

    plt.close()

    # Loss

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Training Loss")

    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.savefig("outputs/training_loss.png")

    plt.close()


def print_header(title):
    """
    Print formatted title.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)