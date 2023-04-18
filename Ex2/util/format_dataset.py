import os.path

from Ex2 import DATASET_PATH
import pandas as pd

image_path = lambda name, image_number: f"{DATASET_PATH}/{name}/{name}_{int(image_number):04d}.jpg"


def format_dataset(pairs_path: str):
    cntsame,cntdiff = [0,0]
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
