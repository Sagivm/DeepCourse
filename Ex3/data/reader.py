import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pretty_midi import PrettyMIDI
from gensim.models import Word2Vec, KeyedVectors
from Ex3 import TRAIN_VECTOR_PATH
import nltk
from gensim import models
import re

# import gensim.models.keyedvectors as word2vec
nltk.download('punkt')
nltk.download('')


def read_midi(path):
    t = PrettyMIDI(path)
    x = 0


def save_model_vectors(path, language_model, tokenized_corpus):
    with open(path, 'wb') as handle:
        vectorize_corpus = [language_model[text] for text in tokenized_corpus]
        pickle.dump(vectorize_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    tokenized_train_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in train_corpus]
    tokenized_test_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in test_corpus]

    stop_words = set(nltk.corpus.stopwords.words('english'))
    pattern = re.compile('[^a-zA-Z]')

    # Remove the stopwords
    tokenized_train_corpus = [token.replace("'s", "") for x in tokenized_train_corpus for token in x if
                              (not token.lower() in stop_words)]
    tokenized_train_corpus = [pattern.sub('', token) for token in tokenized_train_corpus]
    tokenized_train_corpus = [token for x in tokenized_train_corpus for token in x if len(token) != 0 and token != 'a']

    # Remove non-alphabetic characters from the tokens

    tokenized_test_corpus = [token.replace("'s", "") for x in tokenized_test_corpus for token in x if
                             (not token.lower() in stop_words)]
    tokenized_test_corpus = [pattern.sub('', token) for token in tokenized_test_corpus]
    tokenized_test_corpus = [token for x in tokenized_test_corpus for token in x if len(token) != 0 and token != 'a']
    # Remove non-alphabetic characters from the tokens


    # TODO: add fine tune using the corpus

    # language_model.build_vocab(tokenized_train_corpus, update=True)

    # Train the model with the new data
    # language_model.train(tokenized_train_corpus, total_examples=language_model.corpus_count,
    #                      epochs=language_model.epochs)

    save_model_vectors(train_dst_path, language_model, tokenized_train_corpus)
    save_model_vectors(test_dst_path, language_model, tokenized_test_corpus)

#
# read_midi("midi_files/All_4_One_-_I_Swear.mid")
# read_songs()
