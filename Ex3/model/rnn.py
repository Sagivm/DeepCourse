from Ex3 import LMODEL_PATH, MAX_SEQ_LENGTH
# import libraries
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
import pickle
import numpy as np

from gensim.models import KeyedVectors, LdaModel

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import LSTM, concatenate, GRU
from keras.layers import Embedding
from keras.regularizers import L1L2
from keras.utils import pad_sequences
from keras import backend as K


class WeightedDropout(Dropout):
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        # Calculate the L2 norm of the weight matrix
        weights = self.layer.kernel
        weight_norm = K.sqrt(K.sum(K.square(weights)))

        # Scale the dropout rate by the weight norm
        rate = self.rate / (1.0 + weight_norm)

        return super().call(inputs, rate=rate, training=training)



def get_songs(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def define_model(vocabulary_size, embedding_size, embedding_weights, midi_size):
    model_wv = Sequential()

    lyrics_input = Input(shape=(None,), name="lyrics")
    mid_input = Input(shape=(midi_size,), name="mid")

    # embedding layer
    lyrics_features = Embedding(vocabulary_size, embedding_size, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_weights], trainable=True, )(lyrics_input)
    # # lstm layer 1
    # lyrics_features = LSTM(128, return_sequences= True)(lyrics_features)

    # lstm layer 2
    # # when using multiple LSTM layers, set return_sequences to True at the previous layer
    # # because the current layer expects a sequential intput rather than a single input
    lyrics_features = LSTM(64)(lyrics_features)

    mid_features = Dense(64, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(mid_input)

    x = concatenate([lyrics_features, mid_features])
    # output layer
    x = Dense(128, activation='relu')(x)
    #x = WeightedDropout(0.5)(x)
    x = Dense(vocabulary_size, activation='softmax')(x)
    # x = RandomProportionalLayer(vocabulary_size)(x)



    return Model(inputs=[lyrics_input, mid_input], outputs=[x])


def rnn(train_songs_path):
    """

    :param train_songs_path:
    :type train_songs_path:
    :return:
    :rtype:
    """
    # Get filtered text
    train_songs, midis = get_songs(train_songs_path)
    word_tokeniser = Tokenizer()
    word_tokeniser.fit_on_texts(train_songs)
    encoded_songs = word_tokeniser.texts_to_sequences(train_songs)

    # check the size of the vocabulary
    VOCABULARY_SIZE = len(word_tokeniser.word_index) + 1
    print('Vocabulary Size: {}'.format(VOCABULARY_SIZE))

    # Make sequences with MAX_SEQ_LENGTH + 1

    max_midi_len = max([midi.size for midi in midis])
    sequences = []
    midis_by_sequence = []
    for index, sample in enumerate(encoded_songs):
        sample_sequences = []
        for i in range(MAX_SEQ_LENGTH, len(sample)):
            sample_sequence = sample[i - MAX_SEQ_LENGTH:i + 1]
            sample_sequences.append(sample_sequence)
            midis_by_sequence.append(np.pad(midis[index], [(0, max_midi_len - midis[index].size)]))
            if midis_by_sequence[-1].size == 1907161:
                t = 0
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

    midi_size = midis_by_sequence[0].size
    model_wv = define_model(VOCABULARY_SIZE, EMBEDDING_SIZE, embedding_weights, midi_size)

    # compile network
    model_wv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model
    model_wv.summary()

    # fit network
    history = model_wv.fit([X, np.stack(midis_by_sequence)], y, epochs=4, verbose=1, batch_size=64,
                           validation_split=0.1)

    with open("run.dump", 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    x = 0

    return model_wv, word_tokeniser,max_midi_len


# generate a sequence from a language model
def generate_words(test_path, model, word_tokeniser, MAX_SEQ_LENGTH, seed, n_words, max_midi_len):
    text = seed

    test_songs, midis = get_songs(test_path)
    # generate n_words
    midis[0] = np.pad(midis[0], [(0, max_midi_len - midis[0].size)])
    midis[0] = np.reshape(midis[0], (1, len(midis[0])))

    for _ in range(n_words):

        # encode text as integers
        encoded_words = word_tokeniser.texts_to_sequences([text])[0]

        # pad sequences
        padded_words = pad_sequences([encoded_words], maxlen=MAX_SEQ_LENGTH, padding='pre')

        # predict next word
        predict_x = model.predict([padded_words, midis[0]])
        prediction = np.argmax(predict_x, axis=1)
        # prediction = model.predict_classes(padded_words, verbose=0)

        # convert predicted index to its word
        next_word = ""
        for word, i in word_tokeniser.word_index.items():
            if i == prediction:
                next_word = word
                break

        # append predicted word to text
        text += " " + next_word

    print(text)
