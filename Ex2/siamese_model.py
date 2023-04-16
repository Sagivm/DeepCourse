import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D


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
        Dense(4096, activation="sigmoid")
    ])
    return model
