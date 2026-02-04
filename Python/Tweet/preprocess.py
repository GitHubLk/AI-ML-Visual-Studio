import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

CSV_PATH = "../datasets/Amazon-Food-Reviews.csv"
N_SAMPLES = 50000
MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 100          # sequence length for LSTM
TEST_SIZE = 0.3            # 70% train, 30% test
RANDOM_STATE = 42
TOKENIZER_PATH = "tokenizer.pickle"
DATA_PATH = "reviews_data.npz"

def clean_text(text: str) -> str:
    """Simple text cleaner: lowercase, remove URLs, non-letters, extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
    text = re.sub(r"[^a-z\s]", " ", text)            # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()         # collapse spaces
    return text
def main():
    print("Loading dataset...")
    if N_SAMPLES is not None:
        df = pd.read_csv(CSV_PATH, nrows=N_SAMPLES)
    else:
        df = pd.read_csv(CSV_PATH)
    #   Text  -> review text
    #   Score -> rating 1–5 (regression target)
    if "Text" not in df.columns or "Score" not in df.columns:
        raise ValueError("Expected columns 'Text' and 'Score' in Reviews.csv")
    df = df[["Text", "Score"]].dropna()
    print(f"Dataset shape after selecting columns & dropping NaNs: {df.shape}")
    texts = df["Text"].astype(str).apply(clean_text).values
    scores = df["Score"].astype(float).values  # ratings 1–5
    print("Fitting tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    print("Example sequence length:", len(sequences[0]))
    print("Padding sequences...")
    X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    y = scores
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    # Train/test split 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("Train size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])
    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
    print("Vocabulary size (capped):", vocab_size)
    # Save processed data
    print(f"Saving processed data to {DATA_PATH}...")
    np.savez_compressed(DATA_PATH, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, max_seq_len=MAX_SEQ_LEN, vocab_size=vocab_size)
    # Save tokenizer
    print(f"Saving tokenizer to {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print("Preprocessing done.")

if __name__ == "__main__":
    main()