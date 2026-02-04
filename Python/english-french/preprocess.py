import re
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "../datasets/eng_-french.csv"
PROCESSED_PATH = "eng_fr_processed.npz"
SRC_TOKENIZER_PATH = "src_tokenizer.pickle"
TGT_TOKENIZER_PATH = "tgt_tokenizer.pickle"
CONFIG_PATH = "seq2seq_config.json"
N_SAMPLES = 20000          # subset for speed; set None to use all
MAX_NUM_WORDS_SRC = 20000
MAX_NUM_WORDS_TGT = 20000
SOS_TOKEN = "sostok"
EOS_TOKEN = "eostok"

def clean_sentence(s: str) -> str:
    """Basic lowercasing + space normalization."""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    print("Loading dataset...")
    if N_SAMPLES is not None:
        df = pd.read_csv(DATA_PATH, nrows=N_SAMPLES)
    else:
        df = pd.read_csv(DATA_PATH)

    # Assume first two columns are English and French
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (English, French) in the CSV")

    eng_raw = df.iloc[:, 0].astype(str)
    fra_raw = df.iloc[:, 1].astype(str)

    print("Number of sentence pairs loaded:", len(eng_raw))

    # Clean sentences
    eng_sentences = eng_raw.apply(clean_sentence).tolist()
    fra_sentences = fra_raw.apply(clean_sentence).tolist()

    # Add SOS/EOS tokens to French
    fra_sentences_with_tokens = [f"{SOS_TOKEN} {s} {EOS_TOKEN}" for s in fra_sentences]

    print("Fitting source (English) tokenizer...")
    src_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_SRC, oov_token="<OOV>")
    src_tokenizer.fit_on_texts(eng_sentences)

    print("Fitting target (French) tokenizer...")
    tgt_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_TGT, oov_token="<OOV>")
    tgt_tokenizer.fit_on_texts(fra_sentences_with_tokens)

    # Small debug print
    print("\nSample of target word_index:")
    for i, (w, idx) in enumerate(tgt_tokenizer.word_index.items()):
        if i >= 20:
            break
        print(f"  {w!r}: {idx}")
    print("SOS_TOKEN in vocab?", SOS_TOKEN in tgt_tokenizer.word_index)
    print("EOS_TOKEN in vocab?", EOS_TOKEN in tgt_tokenizer.word_index)

    # Text -> sequences
    print("\nConverting texts to sequences...")
    encoder_sequences = src_tokenizer.texts_to_sequences(eng_sentences)
    decoder_sequences = tgt_tokenizer.texts_to_sequences(fra_sentences_with_tokens)

    # Max lengths
    max_encoder_seq_len = max(len(seq) for seq in encoder_sequences)
    max_decoder_seq_len = max(len(seq) for seq in decoder_sequences)

    print("Max encoder seq len:", max_encoder_seq_len)
    print("Max decoder seq len:", max_decoder_seq_len)

    # Pad
    print("Padding sequences...")
    encoder_input_data = pad_sequences(
        encoder_sequences,
        maxlen=max_encoder_seq_len,
        padding="post",
        truncating="post"
    )

    decoder_sequences_padded = pad_sequences(
        decoder_sequences,
        maxlen=max_decoder_seq_len,
        padding="post",
        truncating="post"
    )

    # Create decoder input/target by shifting
    # decoder_input:  all timesteps except last
    # decoder_target: all timesteps except first
    decoder_input_data = decoder_sequences_padded[:, :-1]
    decoder_target_data = decoder_sequences_padded[:, 1:]
    max_decoder_input_len = decoder_input_data.shape[1]

    print("Encoder input shape:", encoder_input_data.shape)
    print("Decoder input shape:", decoder_input_data.shape)
    print("Decoder target shape:", decoder_target_data.shape)

    # Vocab sizes (capped)
    src_vocab_size = min(MAX_NUM_WORDS_SRC, len(src_tokenizer.word_index) + 1)
    tgt_vocab_size = min(MAX_NUM_WORDS_TGT, len(tgt_tokenizer.word_index) + 1)
    print("Source vocab size:", src_vocab_size)
    print("Target vocab size:", tgt_vocab_size)

    # Get SOS/EOS token ids in target vocab
    sos_id = tgt_tokenizer.word_index.get(SOS_TOKEN)
    eos_id = tgt_tokenizer.word_index.get(EOS_TOKEN)

    if sos_id is None or eos_id is None:
        print("Available special tokens keys sample:", list(tgt_tokenizer.word_index.keys())[:30])
        raise ValueError(f"Could not find {SOS_TOKEN} or {EOS_TOKEN} in target tokenizer.word_index.")

    print("SOS token id:", sos_id)
    print("EOS token id:", eos_id)

    # Save processed arrays
    print(f"\nSaving processed data to {PROCESSED_PATH} ...")
    np.savez_compressed(
        PROCESSED_PATH,
        encoder_input_data=encoder_input_data,
        decoder_input_data=decoder_input_data,
        decoder_target_data=decoder_target_data,
        max_encoder_seq_len=max_encoder_seq_len,
        max_decoder_input_len=max_decoder_input_len,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    )

    # Save tokenizers
    print(f"Saving source tokenizer to {SRC_TOKENIZER_PATH} ...")
    with open(SRC_TOKENIZER_PATH, "wb") as f:
        pickle.dump(src_tokenizer, f)

    print(f"Saving target tokenizer to {TGT_TOKENIZER_PATH} ...")
    with open(TGT_TOKENIZER_PATH, "wb") as f:
        pickle.dump(tgt_tokenizer, f)

    # Save config
    config = {
        "max_encoder_seq_len": int(max_encoder_seq_len),
        "max_decoder_input_len": int(max_decoder_input_len),
        "src_vocab_size": int(src_vocab_size),
        "tgt_vocab_size": int(tgt_vocab_size),
        "sos_id": int(sos_id),
        "eos_id": int(eos_id),
        "sos_token": SOS_TOKEN,
        "eos_token": EOS_TOKEN
    }

    print(f"Saving config to {CONFIG_PATH} ...")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Preprocessing done.")


if __name__ == "__main__":
    main()
