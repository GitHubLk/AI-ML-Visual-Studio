import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_PATH = "reviews_data.npz"
MODEL_PATH = "lstm_rating_model.keras"
BATCH_SIZE = 64
EPOCHS = 100

def build_model(vocab_size, max_seq_len):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_seq_len),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        # Regression output: predict rating 1–5
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
def main():
    print(f"Loading processed data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    max_seq_len = int(data["max_seq_len"])
    vocab_size = int(data["vocab_size"])
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
    print("max_seq_len  :", max_seq_len)
    print("vocab_size   :", vocab_size)
    model = build_model(vocab_size, max_seq_len)
    model.summary()
    print("Training model...")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
    print("Evaluating on test data...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.3f} (on rating scale 1–5)")
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    main()