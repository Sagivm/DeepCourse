import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pretty_midi import PrettyMIDI
from gensim.models import Word2Vec, KeyedVectors
from Ex3 import LMODEL_PATH
import nltk
from gensim import models
import re
import os

# import gensim.models.keyedvectors as word2vec
nltk.download('punkt')
nltk.download('')


def read_midi(path):
    try:
        midi_data = PrettyMIDI(path)
        # Get piano roll representation of MIDI data
        # piano_roll = midi_data.get_piano_roll(fs=100)
        #
        # # Convert piano roll to binary matrix (1 if note is played, 0 otherwise)
        # piano_roll_binary = np.where(piano_roll > 0, 1, 0)
        #
        # # Flatten matrix into a 1D vector
        # vector = piano_roll_binary.flatten()
        return midi_data.get_beats()
        # return mid.get_pitch_class_transition_matrix()
    except Exception:
        print(f"INVALID - {path}")
        return np.zeros((1,))


def read_midis(path):
    midis = os.listdir(path)
    return [read_midi(os.path.join(path, midi)) for midi in midis]


def save_songs(path, tokenized_corpus):
    with open(path, 'wb') as handle:
        pickle.dump(tokenized_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


def filter_text(corpus):
    tokenized_corpus = []
    stop_words = set(nltk.corpus.stopwords.words('english'))
    pattern = re.compile('[^a-zA-Z&]')
    # Remove the stopwords
    for sample in corpus:
        x = 0
        # tokenized_sample = [token.replace("'s", "") for token in sample if (not token.lower() in stop_words)]
        tokenized_sample = [token.replace("'s", "") for token in sample]
        tokenized_sample = [pattern.sub('', token) for token in tokenized_sample]
        tokenized_sample = [token for token in tokenized_sample if len(token) != 0 and token != 'a']
        tokenized_corpus.append(tokenized_sample)
    return tokenized_corpus


def get_midi_paths(df, base_path):
    return list(
        map(lambda x, y: f"{base_path}{x.strip().replace(' ', '_')}_-_{y.strip().replace(' ', '_')}.mid", df.iloc[:, 0].tolist(),
            df.iloc[:, 1].tolist()))


def read_songs(train_src_path, test_src_path, train_dst_path, test_dst_path):
    # Read train
    train_df = pd.read_csv(train_src_path)
    train_df = train_df.iloc[:, :3]
    train_corpus = train_df.iloc[:, 2].tolist()

    # Read Test
    test_df = pd.read_csv(test_src_path)
    test_df = test_df.iloc[:, :3]
    test_corpus = test_df.iloc[:, 2].tolist()

    # # TODO: add download for google trained model

    # Filter song midi
    songs_midis_path_train = get_midi_paths(train_df, "data/midi_files/")
    # train_songs_midis = [read_midi(path) for path in songs_midis_path]

    # Filter song midi
    songs_midis_path_test = get_midi_paths(test_df, "data/midi_files/")
    # test_songs_midis = [read_midi(path) for path in songs_midis_path]

    # Filter song lyrics
    # Load Google's pre-trained Word2Vec model.
    tokenized_train_corpus = [nltk.word_tokenize(sentence.lower().replace('-', ' ')) for sentence in train_corpus]
    tokenized_test_corpus = [nltk.word_tokenize(sentence.lower().replace('-', ' ')) for sentence in test_corpus]

    tokenized_train_corpus = filter_text(tokenized_train_corpus)
    tokenized_test_corpus = filter_text(tokenized_test_corpus)

    # TODO: add fine tune using the corpus
    save_songs("data/tokenized_train_text.pkl", [" ".join(i) for i in tokenized_train_corpus])
    save_songs("data/tokenized_test_text.pkl", [" ".join(i) for i in tokenized_test_corpus])
    return (tokenized_train_corpus, songs_midis_path_train), (tokenized_test_corpus, songs_midis_path_test)



#
# read_midis("data/midi_files/")
# # read_songs()
