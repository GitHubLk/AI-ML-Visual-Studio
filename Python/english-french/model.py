# train_eng_fr_seq2seq.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

PROCESSED_PATH = "eng_fr_processed.npz"
CONFIG_PATH = "seq2seq_config.json"

FULL_MODEL_PATH = "eng_fr_seq2seq_full.keras"
ENCODER_MODEL_PATH = "eng_fr_encoder.keras"
DECODER_MODEL_PATH = "eng_fr_decoder.keras"


def build_train_model(src_vocab_size, tgt_vocab_size, latent_dim, max_decoder_input_len):
    # Encoder
    encoder_inputs = layers.Input(shape=(None,), name="encoder_inputs")
    enc_emb = layers.Embedding(input_dim=src_vocab_size, output_dim=128, mask_zero=True, name="encoder_embedding")(encoder_inputs)
    encoder_lstm = layers.LSTM(latent_dim, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]
    # Decoder
    decoder_inputs = layers.Input(shape=(None,), name="decoder_inputs")
    dec_emb_layer = layers.Embedding(input_dim=tgt_vocab_size, output_dim=128, mask_zero=True, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = layers.Dense(tgt_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model")
    return model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense

def build_inference_models(encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense, latent_dim):
    # Encoder model: input English sequence -> states
    encoder_model = Model(encoder_inputs, encoder_states, name="encoder_model")
    # Decoder model:
    # Inputs:
    #   1) current target token
    #   2) previous state_h
    #   3) previous state_c
    decoder_state_input_h = layers.Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = layers.Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2, name="decoder_model")
    return encoder_model, decoder_model

def main():
    print("Loading processed data...")
    data = np.load(PROCESSED_PATH)

    encoder_input_data = data["encoder_input_data"]
    decoder_input_data = data["decoder_input_data"]
    decoder_target_data = data["decoder_target_data"]
    max_encoder_seq_len = int(data["max_encoder_seq_len"])
    max_decoder_input_len = int(data["max_decoder_input_len"])
    src_vocab_size = int(data["src_vocab_size"])
    tgt_vocab_size = int(data["tgt_vocab_size"])

    print("Encoder input shape:", encoder_input_data.shape)
    print("Decoder input shape:", decoder_input_data.shape)
    print("Decoder target shape:", decoder_target_data.shape)
    print("max_encoder_seq_len:", max_encoder_seq_len)
    print("max_decoder_input_len:", max_decoder_input_len)
    print("src_vocab_size:", src_vocab_size)
    print("tgt_vocab_size:", tgt_vocab_size)

    latent_dim = 256

    # Build training model
    model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense = \
        build_train_model(src_vocab_size, tgt_vocab_size, latent_dim, max_decoder_input_len)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # decoder_target_data should be (samples, timesteps)
    # Keras will broadcast it for sparse_categorical_crossentropy
    BATCH_SIZE = 64
    EPOCHS = 10

    print("Training seq2seq model...")
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1
    )

    print(f"Saving full training model to {FULL_MODEL_PATH} ...")
    model.save(FULL_MODEL_PATH)

    # Build inference models using the trained layers
    print("Building inference encoder/decoder models...")
    encoder_model, decoder_model = build_inference_models(
        encoder_inputs,
        encoder_states,
        decoder_inputs,
        dec_emb_layer,
        decoder_lstm,
        decoder_dense,
        latent_dim
    )

    print(f"Saving encoder model to {ENCODER_MODEL_PATH} ...")
    encoder_model.save(ENCODER_MODEL_PATH)

    print(f"Saving decoder model to {DECODER_MODEL_PATH} ...")
    decoder_model.save(DECODER_MODEL_PATH)

    print("Training and saving done.")


if __name__ == "__main__":
    main()