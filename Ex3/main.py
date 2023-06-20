import tensorflow as tf
import numpy as np
import os
import time
from Ex3 import LMODEL_PATH, TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH, MAX_SEQ_LENGTH
from Ex3.data.pre_process import read_songs
from Ex3.model.rnn import rnn, generate_words
from Ex3.data import notes_to_lyrics

#read_songs(TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH)
model_wv, word_tokeniser, max_midi_len = rnn(TRAIN_VECTOR_PATH)

seed_text = "close your eyes give"
num_words = 100
print(generate_words(TSET_VECTOR_PATH, model_wv, word_tokeniser, MAX_SEQ_LENGTH, seed_text, num_words, max_midi_len))
