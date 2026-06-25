import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully!")
    print("Shape:", df.shape)

    columns_to_drop = [
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Timestamp",
        "Label.1"
    ]

    df.drop(columns=columns_to_drop,
            errors="ignore",
            inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_train = X_train.reshape(
        X_train.shape[0],
        1,
        X_train.shape[1]
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        1,
        X_test.shape[1]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoder
    )