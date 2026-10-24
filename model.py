import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed,
    Attention,
    Add,
    GlobalAveragePooling1D
)


def build_initial_autoencoder(
    sequence_length=60,
    n_features=6
):
    inputs = Input(
        shape=(sequence_length, n_features),
        name="layer1_input"
    )

    x = LSTM(
        64,
        return_sequences=True,
        name="layer1_encoder_lstm_1"
    )(inputs)

    x = Dropout(
        0.2,
        name="layer1_dropout"
    )(x)

    x = LSTM(
        32,
        return_sequences=False,
        name="layer1_encoder_lstm_2"
    )(x)

    latent = Dense(
        16,
        activation="linear",
        name="layer1_latent"
    )(x)

    x = RepeatVector(
        sequence_length,
        name="layer1_repeat"
    )(latent)

    x = LSTM(
        32,
        return_sequences=True,
        name="layer1_decoder_lstm_1"
    )(x)

    x = Dropout(
        0.2,
        name="layer1_decoder_dropout"
    )(x)

    x = LSTM(
        64,
        return_sequences=True,
        name="layer1_decoder_lstm_2"
    )(x)

    outputs = TimeDistributed(
        Dense(n_features),
        name="layer1_reconstruction"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="Layer1_LSTM_Autoencoder"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mse"
    )

    return model


def build_temporal_attention_autoencoder(
    sequence_length=60,
    n_features=6
):
    inputs = Input(
        shape=(sequence_length, n_features),
        name="layer2_input"
    )

    x = LSTM(
        64,
        return_sequences=True,
        name="layer2_encoder_lstm_1"
    )(inputs)

    x = Dropout(
        0.2,
        name="layer2_dropout"
    )(x)

    x = LSTM(
        32,
        return_sequences=True,
        name="layer2_encoder_lstm_2"
    )(x)

    attention = Attention(
        name="layer2_self_attention"
    )([x, x])

    x = Add(
        name="layer2_attention_residual"
    )([x, attention])

    x = GlobalAveragePooling1D(
        name="layer2_global_pool"
    )(x)

    latent = Dense(
        16,
        activation="linear",
        name="layer2_latent"
    )(x)

    x = RepeatVector(
        sequence_length,
        name="layer2_repeat"
    )(latent)

    x = LSTM(
        32,
        return_sequences=True,
        name="layer2_decoder_lstm_1"
    )(x)

    x = Dropout(
        0.2,
        name="layer2_decoder_dropout"
    )(x)

    x = LSTM(
        64,
        return_sequences=True,
        name="layer2_decoder_lstm_2"
    )(x)

    outputs = TimeDistributed(
        Dense(n_features),
        name="layer2_reconstruction"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="Layer2_LSTM_Attention_Autoencoder"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mse"
    )

    return model
