import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

'''
    3D VQVAE parts for animation
'''


def get_animation_encoder(latent_dim, num_frames):
    encoder_inputs = keras.Input(shape=(num_frames, 128, 128, 3), name="input")
    x = layers.Conv3D(32, 3, activation="relu", strides=(1, 2, 2), padding="same")(
        encoder_inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, 3, activation="relu", strides=(1, 2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, 1, activation="relu", strides=1, padding="same")(x)
    encoder_outputs = layers.Conv3D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_animation_decoder(latent_dim, num_frames):
    latent_inputs = keras.Input(shape=get_animation_encoder(latent_dim, num_frames).output.shape[1:])

    x = layers.Conv3DTranspose(128, 3, activation="relu", strides=(1, 2, 2), padding="same")(
        latent_inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Conv3DTranspose(64, 3, activation="relu", strides=(1, 2, 2), padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Conv3DTranspose(64, 3, activation="relu", strides=1, padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Conv3DTranspose(64, 3, activation="relu", strides=1, padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Conv3DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3DTranspose(16, 1, activation="relu", strides=1, padding="same")(x)
    decoder_outputs = layers.Conv3DTranspose(3, 3, padding="same", activation="sigmoid", name="output")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def get_animation_vqvae(latent_dim=32, num_embeddings=64, num_frames=6):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_animation_encoder(latent_dim, num_frames)
    decoder = get_animation_decoder(latent_dim, num_frames)
    inputs = keras.Input(shape=(num_frames, 128, 128, 3))

    # TODO: reintroduce augmentation
    # augmented = keras.layers.RandomFlip('horizontal')(inputs)

    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    model = keras.Model(inputs, reconstructions, name="vq_vae")
    print(model.summary())
    return model


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
