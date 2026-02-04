import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "reviews_data.npz"
TOKENIZER_PATH = "tokenizer.pickle"
MODEL_PATH = "lstm_rating_model.keras"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
    text = re.sub(r"[^a-z\s]", " ", text)            # only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
def load_artifacts():
    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("Loading config from processed data...")
    data = np.load(DATA_PATH)
    max_seq_len = int(data["max_seq_len"])
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    return tokenizer, max_seq_len, model
def predict_rating(model, tokenizer, max_seq_len, text: str) -> float:
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_seq_len, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0, 0]  # scalar
    # Clip to [1, 5] for sanity
    pred = float(np.clip(pred, 1.0, 5.0))
    return pred
def generate_reply(rating: float) -> str:
    """Tiny 'chatbot brain' based on predicted rating."""
    if rating >= 4.5:
        return f"Wow, that sounds super positive! I'd rate your vibe {rating:.1f}/5 🌟"
    elif rating >= 3.5:
        return f"Seems like you mostly liked it. I'd say about {rating:.1f}/5 🙂"
    elif rating >= 2.5:
        return f"Mixed feelings detected... around {rating:.1f}/5 😐"
    elif rating >= 1.5:
        return f"Oof, that doesn’t sound great. Maybe {rating:.1f}/5 😕"
    else:
        return f"Yikes, that review is brutal. I'd give it about {rating:.1f}/5 😬"
def main():
    tokenizer, max_seq_len, model = load_artifacts()
    print("\nWelcome to the Rate-My-Review Bot!")
    print("Type anything that looks like a product review.")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Bot: Bye! 🖐")
            break
        if not user_text:
            continue
        rating = predict_rating(model, tokenizer, max_seq_len, user_text)
        reply = generate_reply(rating)
        print(f"Bot: {reply}")
        
if __name__ == "__main__":
    main()
