import os.path

# from Ex2 import DATASET_PATH
# import pandas as pd
import tensorflow as tf

# image_path = lambda name, image_number: f"{DATASET_PATH}/{name}/{name}_{int(image_number):04d}.jpg"
image_path = lambda name, image_number: f"lfw2/lfw2/{name}/{name}_{int(image_number):04d}.jpg"


def create_dataset(pairs):
    def _generator():
        for ((path_a, path_b), label) in pairs:
            yield (tf.keras.utils.load_img(path_a), tf.keras.utils.load_img(path_b)), label

    return tf.data.Dataset.from_generator(_generator,
                                          output_signature=((tf.TensorSpec((250, 250, 3), tf.uint8),
                                                             tf.TensorSpec((250, 250, 3), tf.uint8)),
                                                            tf.TensorSpec((), tf.uint8)))


def format_dataset(pairs_path: str):
    with open(pairs_path) as file:
        content = file.readlines()
    parsed_paires = [line[:-1].split("\t") for line in content]
    pairs = list()
    for pair in parsed_paires:
        if len(pair) == 3:  # Same name so same person
            pairs.append(((
                image_path(pair[0], pair[1]),
                image_path(pair[0], pair[2])),
                1
            ))

        elif len(pair) == 4:  # Different name so different person
            pairs.append(((
                image_path(pair[0], pair[1]),
                image_path(pair[2], pair[3])),
                0
            ))
    return pairs

