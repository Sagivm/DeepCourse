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
    mid = PrettyMIDI(path)
    return mid.get_pitch_class_transition_matrix()


def read_midis(path):
    midis = os.listdir(path)
    return [read_midi(os.path.join(path, midi)) for midi in midis]


# Used get mean vector for non english phrases, the result for them whould be a 0 vector
def save_model_vectors(path, language_model, tokenized_corpus):
    max_len = max([len(song) for song in tokenized_corpus])
    with open(path, 'wb') as handle:
        vectorize_corpus = []
        for sample in tokenized_corpus:
            tmp = np.stack([language_model.get_mean_vector(text) for text in sample])
            tmp = np.pad(tmp, ((0, max_len - tmp.shape[0]), (0, 0)))
            vectorize_corpus.append(tmp)
        tokenized_corpus = [" ".join(sample) for sample in tokenized_corpus]
        pickle.dump((tokenized_corpus, vectorize_corpus), handle, protocol=pickle.HIGHEST_PROTOCOL)


def filter_text(corpus):
    tokenized_corpus = []
    stop_words = set(nltk.corpus.stopwords.words('english'))
    pattern = re.compile('[^a-zA-Z]')
    # Remove the stopwords
    for sample in corpus:
        x = 0
        tokenized_sample = [token.replace("'s", "") for token in sample if (not token.lower() in stop_words)]
        tokenized_sample = [pattern.sub('', token) for token in tokenized_sample]
        tokenized_sample = [token for token in tokenized_sample if len(token) != 0 and token != 'a']
        tokenized_corpus.append(tokenized_sample)
    return tokenized_corpus


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

    # Load Google's pre-trained Word2Vec model.
    language_model = KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
    tokenized_train_corpus = [nltk.word_tokenize(sentence.lower().replace('-', ' ')) for sentence in train_corpus]
    tokenized_test_corpus = [nltk.word_tokenize(sentence.lower().replace('-', ' ')) for sentence in test_corpus]

    tokenized_train_corpus = filter_text(tokenized_train_corpus)
    tokenized_test_corpus = filter_text(tokenized_test_corpus)

    # TODO: add fine tune using the corpus

    save_model_vectors(train_dst_path, language_model, tokenized_train_corpus)
    save_model_vectors(test_dst_path, language_model, tokenized_test_corpus)

    # save model
    with open(LMODEL_PATH, 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

#
# read_midis("data/midi_files/")
# # read_songs()
