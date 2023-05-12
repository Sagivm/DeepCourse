from Ex3 import LMODEL_PATH, MAX_SEQ_LENGTH
# import libraries
import warnings

warnings.filterwarnings("ignore")
import pickle
import numpy as np

from gensim.models import KeyedVectors, LdaModel

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


def get_songs(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def define_model(vocabulary_size, embedding_size, embedding_weights):
    # create model architecture

    model_wv = Sequential()

    # embedding layer
    model_wv.add(Embedding(vocabulary_size, embedding_size, input_length=MAX_SEQ_LENGTH,
                           weights=[embedding_weights], trainable=True))

    # lstm layer 1
    model_wv.add(LSTM(128, return_sequences=True))

    # lstm layer 2
    # when using multiple LSTM layers, set return_sequences to True at the previous layer
    # because the current layer expects a sequential intput rather than a single input
    model_wv.add(LSTM(128))

    # output layer
    model_wv.add(Dense(vocabulary_size, activation='softmax'))

    return model_wv


def rnn(train_songs_path):
    """

    :param train_songs_path:
    :type train_songs_path:
    :return:
    :rtype:
    """
    # Get filtered text
    train_songs = get_songs(train_songs_path)
    word_tokeniser = Tokenizer()
    word_tokeniser.fit_on_texts(train_songs)
    encoded_songs = word_tokeniser.texts_to_sequences(train_songs)

    # check the size of the vocabulary
    VOCABULARY_SIZE = len(word_tokeniser.word_index) + 1
    print('Vocabulary Size: {}'.format(VOCABULARY_SIZE))

    # Make sequences with MAX_SEQ_LENGTH + 1

    sequences = []
    for sample in encoded_songs:
        sample_sequences = []
        for i in range(MAX_SEQ_LENGTH, len(sample)):
            sample_sequence = sample[i - MAX_SEQ_LENGTH:i + 1]
            sample_sequences.append(sample_sequence)
        sequences.append(np.array(sample_sequences))
    sequences = np.vstack(sequences)

    # divide the sequence into X and y
    X = sequences[:, :-1]  # assign all but last words of a sequence to X
    y = sequences[:, -1]  # assign last word of each sequence to
    y = to_categorical(y, num_classes=VOCABULARY_SIZE)


    # load word2vec using the following function present in the gensim library
    word2vec = KeyedVectors.load_word2vec_format(LMODEL_PATH, binary=True)

    # assign word vectors from word2vec model

    EMBEDDING_SIZE = 300  # each word in word2vec model is represented using a 300 dimensional vector

    # create an empty embedding matrix
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))

    # create a word to index dictionary mapping
    word2id = word_tokeniser.word_index

    # copy vectors from word2vec model to the words present in corpus
    for word, index in word2id.items():
        try:
            embedding_weights[index, :] = word2vec[word]
        except KeyError:
            pass

    model_wv = define_model(VOCABULARY_SIZE, EMBEDDING_SIZE, embedding_weights)

    # compile network
    model_wv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model
    model_wv.summary()

    # fit network
    history = model_wv.fit(X, y, epochs=2, verbose=1, batch_size=256)

    x = 0
