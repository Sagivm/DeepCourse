import numpy as np
import pickle
import tensorflow as tf
from keras import layers,models
from keras.layers import Input,Dense,Concatenate
from Ex4 import TRAIN_VECTOR_PATH

def get_songs(path):
    with open(path, 'rb') as handle:
        t: dict = pickle.load(handle)
        return t

# Generator model
def build_generator_model(embedding_length,mid_dim, noise_dim=16):
    """
    Define generator model which gets an input of desired output embedding length , mid dim and noise dim
    :param embedding_length:
    :type embedding_length:
    :param mid_dim:
    :type mid_dim:
    :param noise_dim:
    :type noise_dim:
    :return:
    :rtype:
    """
    model = tf.keras.Sequential()

    # Dense layer with input vector size of 128 + random noise
    model.add(layers.Dense(256, input_shape=(mid_dim + noise_dim,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    # Dense layer
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    # Dense layer
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    # Output layer with embedding length
    model.add(layers.Dense(embedding_length, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# Define the discriminator
def build_discriminator_model(song_input_size,melody_input_size):
    """
    Build disciminator model which gets the size of a song embedding and the size of the melody
    :param song_input_size:
    :type song_input_size:
    :param melody_input_size:
    :type melody_input_size:
    :return:
    :rtype:
    """
    input_shape_1 = (song_input_size,)
    input_shape_2 = (song_input_size,)
    input_shape_3 = (melody_input_size,)

    # Discriminator input layers
    input_1 = Input(shape=input_shape_1)
    input_2 = Input(shape=input_shape_2)
    input_3 = Input(shape=input_shape_3)

    # Concatenate the inputs
    merged_inputs = Concatenate()([input_1, input_2, input_3])

    # Shared hidden layers
    x = Dense(128, activation='relu')(merged_inputs)
    x = Dense(64, activation='relu')(x)

    # Output layer
    output = Dense(1, activation='sigmoid')(x)

    # Discriminator model
    discriminator = models.Model(inputs=[input_1, input_2, input_3], outputs=output)
    discriminator.summary()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator
