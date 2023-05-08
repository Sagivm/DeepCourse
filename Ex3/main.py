import tensorflow as tf

import numpy as np
import os
import time
from Ex3 import TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH
from Ex3.data.reader import read_songs

read_songs(TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH)
