from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


def build_model(input_shape, num_classes):
    """
    Builds and compiles the LSTM model.

    Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features)
        num_classes (int): Number of output classes

    Returns:
        model: Compiled Keras model
    """

    model = Sequential([
        Input(shape=input_shape),

        LSTM(
            units=64,
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(
            units=32
        ),

        Dropout(0.2),

        Dense(
            units=64,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            units=num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model