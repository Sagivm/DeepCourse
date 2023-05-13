import tensorflow as tf
import numpy as np
import os
import time
from Ex3 import LMODEL_PATH, TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH
from Ex3.data.pre_process import read_songs
from Ex3.model.rnn import rnn

# read_songs(TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH)
rnn(TRAIN_VECTOR_PATH)
