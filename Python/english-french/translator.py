# translate_eng_fr.py

import re
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

SRC_TOKENIZER_PATH = "src_tokenizer.pickle"
TGT_TOKENIZER_PATH = "tgt_tokenizer.pickle"
CONFIG_PATH = "seq2seq_config.json"

ENCODER_MODEL_PATH = "eng_fr_encoder.keras"
DECODER_MODEL_PATH = "eng_fr_decoder.keras"


def clean_sentence(s: str) -> str:
    """Same basic cleaning as preprocessing."""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_artifacts():
    print("Loading config...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    max_encoder_seq_len = config["max_encoder_seq_len"]
    max_decoder_input_len = config["max_decoder_input_len"]
    sos_id = config["sos_id"]
    eos_id = config["eos_id"]

    print("Loading tokenizers...")
    with open(SRC_TOKENIZER_PATH, "rb") as f:
        src_tokenizer = pickle.load(f)

    with open(TGT_TOKENIZER_PATH, "rb") as f:
        tgt_tokenizer = pickle.load(f)

    print("Loading encoder/decoder models...")
    encoder_model = tf.keras.models.load_model(ENCODER_MODEL_PATH)
    decoder_model = tf.keras.models.load_model(DECODER_MODEL_PATH)

    # Build index->word mapping for French
    index_to_word = {idx: word for word, idx in tgt_tokenizer.word_index.items()}

    return (src_tokenizer, tgt_tokenizer, index_to_word,
            encoder_model, decoder_model,
            max_encoder_seq_len, max_decoder_input_len,
            sos_id, eos_id)


def decode_sequence(input_text,
                    src_tokenizer, index_to_word,
                    encoder_model, decoder_model,
                    max_encoder_seq_len, max_decoder_input_len,
                    sos_id, eos_id):
    # Clean & tokenize English
    cleaned = clean_sentence(input_text)
    seq = src_tokenizer.texts_to_sequences([cleaned])
    encoder_input = pad_sequences(
        seq,
        maxlen=max_encoder_seq_len,
        padding="post",
        truncating="post"
    )

    # Encode input to get initial states
    states_value = encoder_model.predict(encoder_input, verbose=0)

    # Generate French sequence
    target_seq = np.array([[sos_id]], dtype="int32")  # start with <sos>

    stop_condition = False
    decoded_tokens = []

    # We won't go longer than max_decoder_input_len
    while not stop_condition and len(decoded_tokens) < max_decoder_input_len:
        # decoder_model inputs: [decoder_inputs] + state_h + state_c
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0
        )

        # output_tokens shape: (1, 1, tgt_vocab_size)
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        if sampled_token_index == eos_id or sampled_token_index == 0:
            stop_condition = True
        else:
            sampled_word = index_to_word.get(sampled_token_index, "")
            if sampled_word not in {"<sos>", "<eos>", ""}:
                decoded_tokens.append(sampled_word)

        # Update target_seq (the next input token)
        target_seq = np.array([[sampled_token_index]], dtype="int32")

        # Update states
        states_value = [h, c]

    return " ".join(decoded_tokens)


def main():
    (src_tokenizer, tgt_tokenizer, index_to_word,
     encoder_model, decoder_model,
     max_encoder_seq_len, max_decoder_input_len,
     sos_id, eos_id) = load_artifacts()

    print("\nEnglish → French Translator (Seq2Seq LSTM)")
    print("Type an English sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        eng = input("You (EN): ").strip()
        if eng.lower() in {"quit", "exit"}:
            print("Bye 👋")
            break
        if not eng:
            continue

        fr = decode_sequence(
            eng,
            src_tokenizer, index_to_word,
            encoder_model, decoder_model,
            max_encoder_seq_len, max_decoder_input_len,
            sos_id, eos_id
        )

        print("Bot (FR):", fr)
        print()


if __name__ == "__main__":
    main()
