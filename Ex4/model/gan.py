import numpy as np
import pickle
import tensorflow as tf
from keras import layers
from Ex4 import TRAIN_VECTOR_PATH

def get_songs(path):
    with open(path, 'rb') as handle:
        t: dict = pickle.load(handle)
        return t

# Generator model
def make_generator_model(embedding_length, noise_dim=16):
    model = tf.keras.Sequential()

    # Dense layer with input vector size of 128 + random noise
    model.add(layers.Dense(256, input_shape=(128 + noise_dim,)))
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

    return model

# Define the embedding length
embedding_length = 256

# Random noise dimension
noise_dim = 16

# Create an instance of the generator model
generator = make_generator_model(embedding_length)

# Print the model summary
generator.summary()



train,test = get_songs(TRAIN_VECTOR_PATH).values()
mid_encoding = list()
for sample in train:
    sample_mid = [x[1] for x in sample]
    noise = np.random.normal(0, 1, (noise_dim))
    mid_encoding.append(np.concatenate((np.average(sample_mid, axis=0),noise)) )


generator.predict(np.array(mid_encoding))
x=0