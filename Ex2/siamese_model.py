import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten


def build_base_model(input_shape):
    model = tf.keras.Sequential([
        Input(input_shape),
        Conv2D(64, (10, 10), padding="valid", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (7, 7), padding="valid", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (4, 4), padding="valid", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(256, (4, 4), padding="valid", activation="relu"),
        Flatten(),
        Dense(1048, activation="sigmoid")
    ])
    return model


def build_siamese_model(input_shape):
    x1 = Input(shape=input_shape)
    x2 = Input(shape=input_shape)

    base_model = build_base_model(input_shape)
    x1_emb = base_model(x1)
    x2_emb = base_model(x2)

    diff = tf.math.abs(x1_emb - x2_emb)
    final_layer = Dense(1, 'sigmoid')(diff)

    siamese_model = Model(inputs=[x1, x2], outputs=final_layer, name='SiameseModel')

    return siamese_model


def batch_generator(pairs, batch_size):
    while True:
        indices = tf.range(len(pairs))
        tf.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_pairs = [pairs[j] for j in batch_indices]
            batch_a = [np.asarray(imageio.imread(pair[0][0])) for pair in batch_pairs]
            batch_b = [np.asarray(imageio.imread(pair[0][1])) for pair in batch_pairs]
            batch_labels = [pair[1] for pair in batch_pairs]
            yield [tf.stack(batch_a), tf.stack(batch_b)], tf.stack(batch_labels)
