import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from skimage.transform import rotate, AffineTransform, warp
import random


def build_base_model(input_shape):
    """
    Build base model based to create a feature vector of an image
    :param input_shape: image dimensions
    :type input_shape:
    :return:
    :rtype:
    """
    model = tf.keras.Sequential([
        Input(input_shape),
        Conv2D(64, (10, 10), padding="valid", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (7, 7), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
               kernel_regularizer=l2(2e-4)),
        Dropout(0.2),
        MaxPooling2D((2, 2)),
        Conv2D(128, (4, 4), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
               kernel_regularizer=l2(2e-4)),
        Dropout(0.2),
        MaxPooling2D((2, 2)),
        Conv2D(256, (4, 4), padding="valid", activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
               kernel_regularizer=l2(2e-4)),
        Dropout(0.2),
        Flatten(),
        Dense(2048, activation="sigmoid")
    ])
    return model


def build_siamese_model(input_shape):
    """
    Build the model with combined base models
    :param input_shape:
    :type input_shape:
    :return:
    :rtype:
    """
    x1 = Input(shape=input_shape)
    x2 = Input(shape=input_shape)

    base_model = build_base_model(input_shape)
    x1_emb = base_model(x1)
    x2_emb = base_model(x2)

    diff = tf.math.abs(x1_emb - x2_emb)
    final_layer = Dense(1, 'sigmoid')(diff)

    siamese_model = Model(inputs=[x1, x2], outputs=final_layer, name='SiameseModel')

    return siamese_model


def batch_generator(pairs, batch_size, with_transforms=True):
    while True:
        indices = tf.range(len(pairs))
        tf.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_pairs = [pairs[j] for j in batch_indices]
            batch_a = [np.asarray(tf.keras.utils.load_img(pair[0][0], color_mode='grayscale')) for pair in batch_pairs]
            batch_b = [np.asarray(tf.keras.utils.load_img(pair[0][1], color_mode='grayscale')) for pair in batch_pairs]
            batch_labels = [pair[1] for pair in batch_pairs]
            # batch_a = [mat / 256 for mat in batch_a]
            # batch_b = [mat / 256 for mat in batch_b]
            # Prepeocessing
            if with_transforms:
                batch_a += [transform(image) for image in batch_a for _ in range(2)]
                batch_b += [transform(image) for image in batch_b for _ in range(2)]
                batch_labels += batch_labels * 2
            yield [tf.stack(batch_a), tf.stack(batch_b)], tf.stack(batch_labels)


# def test_batch_generator(pairs, batch_size):
#     indices = tf.range(len(pairs))
#     tf.random.shuffle(indices)
#     for i in range(0, len(indices), batch_size):
#         batch_indices = indices[i:i + batch_size]
#         batch_pairs = [pairs[j] for j in batch_indices]
#         batch_a = [np.asarray(imageio.imread(pair[0][0])) for pair in batch_pairs]
#         batch_b = [np.asarray(imageio.imread(pair[0][1])) for pair in batch_pairs]
#
#         # batch_a = [mat / 256 for mat in batch_a]
#         # batch_b = [mat / 256 for mat in batch_b]
#         batch_labels = [pair[1] for pair in batch_pairs]
#         yield [tf.stack(batch_a), tf.stack(batch_b)], tf.stack(batch_labels)


def affinetransform(image):
    transform = AffineTransform(translation=(-30,0))
    warp_image = warp(image,transform, mode="wrap")
    return warp_image


def anticlockwise_rotation(image):
    angle= random.randint(0,45)
    return rotate(image, angle)


def clockwise_rotation(image):
    angle= random.randint(0,45)
    return rotate(image, -angle)


def transform(image):
    if random.random() > 0.5:
        image = affinetransform(image)
    if random.random() > 0.5:
        image = anticlockwise_rotation(image)
    if random.random() > 0.5:
        image = clockwise_rotation(image)

    return image
