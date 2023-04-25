import numpy as np
# import imageio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
# from s.transform import rotate, AffineTransform, warp, rescale
import random

def build_base_model(input_shape):
    W_init_1 = RandomNormal(mean=0, stddev=0.01)
    b_init = RandomNormal(mean=0.5, stddev=0.01)
    W_init_2 = RandomNormal(mean=0, stddev=0.2)
    model = tf.keras.Sequential([
        Input(input_shape),
        Conv2D(64, (10, 10), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)),
        MaxPooling2D(strides=2),
        BatchNormalization(),
        Conv2D(128, (7, 7), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)),
        MaxPooling2D(strides=2),
        BatchNormalization(),
        Conv2D(128, (4, 4), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)),
        MaxPooling2D(strides=2),
        BatchNormalization(),
        Conv2D(128, (4, 4), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)),
        MaxPooling2D(strides=2),
        BatchNormalization(),
        Conv2D(256, (4, 4), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)),
        MaxPooling2D(strides=2),
        BatchNormalization(),
        Conv2D(256, (4, 4), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)),
        Flatten(),
        Dense(2048, activation="sigmoid")
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
            batch_a = [tf.keras.utils.load_img(pair[0][0]) for pair in batch_pairs]
            batch_b = [tf.keras.utils.load_img(pair[0][1]) for pair in batch_pairs]
            batch_labels = [pair[1] for pair in batch_pairs]
            yield [tf.stack(batch_a), tf.stack(batch_b)], tf.stack(batch_labels)



# def affinetransform(image):
#     transform = AffineTransform(translation=(-30,0))
#     warp_image = warp(image,transform, mode="wrap")
#     return warp_image
#
# def anticlockwise_rotation(image):
#     angle= random.randint(0,45)
#     return rotate(image, angle)
#
# def clockwise_rotation(image):
#     angle= random.randint(0,45)
#     return rotate(image, -angle)
#
#
# def transform(image):
#     if random.random() > 0.5:
#         image = affinetransform(image)
#     if random.random() > 0.5:
#         image = anticlockwise_rotation(image)
#     if random.random() > 0.5:
#         image = clockwise_rotation(image)
#
#     return image