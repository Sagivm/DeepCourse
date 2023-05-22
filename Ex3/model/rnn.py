import numpy

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
from keras.layers import LSTM, concatenate, GRU, Add
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
        t: dict = pickle.load(handle)
        return t.values()


# def define_model(vocabulary_size, embedding_size, embedding_weights):
#     model_wv = Sequential()
#
#     lyrics_input = Input(shape=(None, MAX_SEQ_LENGTH,), name="lyrics")
#     mid_input = Input(shape=(None, 128), name="mid")
#
#     # embedding layer
#     lyrics_features = Embedding(vocabulary_size, embedding_size, input_length=MAX_SEQ_LENGTH,
#                                 weights=[embedding_weights], trainable=False)(lyrics_input)
#     mid_features = Dense(300, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(mid_input)
#     # # lstm layer 1
#     # lyrics_features = LSTM(128, return_sequences= True)(lyrics_features)
#     added = Add()([lyrics_features, mid_features])
#     # lstm layer 2
#     # # when using multiple LSTM layers, set return_sequences to True at the previous layer
#     # # because the current layer expects a sequential intput rather than a single input
#     x = LSTM(128, return_sequences=True)(added)
#     x = LSTM(128)(x)
#     # mid_features = Dense(256, activation='relu',)(mid_input)
#     #
#     # x = concatenate([lyrics_features, mid_features])
#     # # output layer
#     x = Dense(256, activation='relu')(x)
#     # x = WeightedDropout(0.5)(x)
#     x = Dense(vocabulary_size, activation='softmax')(x)
#     # x = RandomProportionalLayer(vocabulary_size)(x)
#
#     return Model(inputs=[lyrics_input, mid_input], outputs=[x])

class CustomModel(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size, embedding_weights):
        super().__init__(self)
        self.lyrics_embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size,
                                                          weights=[embedding_weights], mask_zero=True)
        self.midi_embedding = tf.keras.layers.Dense(embedding_size)
        self.gru = tf.keras.layers.GRU(256, return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocabulary_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        words, midis = inputs

        w_emb = self.lyrics_embedding(words, training=training)
        mask = self.lyrics_embedding.compute_mask(words)
        m_emb = self.midi_embedding(midis, training=training)
        x = w_emb + m_emb

        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training, mask=mask)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def split_input_target(words, midis):
    return (words[:-1], midis[:-1]), words[1:]


def rnn(train_songs_path):
    """

    :param train_songs_path:
    :type train_songs_path:
    :return:
    :rtype:
    """
    # Get filtered text
    train_songs, _ = get_songs(train_songs_path)
    word_tokeniser = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
    songs_texts = []
    for song in train_songs:
        songs_texts.append(" ".join([word[0] for word in song]))

    word_tokeniser.fit_on_texts(songs_texts)
    encoded_songs = word_tokeniser.texts_to_sequences(songs_texts)

    max_midi_len = 0
    # retrun encoding to train_songs:
    for i in range(len(encoded_songs)):
        encoded_song = encoded_songs[i]
        train_song = train_songs[i]

        for j in range(len(encoded_song)):
            max_midi_len = max(max_midi_len, len(train_song[j][1]))
            train_song[j] = (encoded_song[j], train_song[j][1])
    # check the size of the vocabulary
    VOCABULARY_SIZE = len(word_tokeniser.word_index) + 1
    print('Vocabulary Size: {}'.format(VOCABULARY_SIZE))

    # Make sequences with MAX_SEQ_LENGTH + 1
    sequences = None
    max_sample_len = max(map(len, train_songs))
    for sample in train_songs:
        words, midis = list(map(lambda x: x[0], sample)), list(map(lambda x: x[1], sample))
        words = np.pad(words, (0, max_sample_len - len(words))).astype(np.int32)
        midis = np.pad(midis, ((0, max_sample_len - len(midis)), (0, 0))).astype(np.int32)

        words_ds = tf.data.Dataset.from_tensor_slices(words)
        midis_ds = tf.data.Dataset.from_tensor_slices(midis)

        sample_ds = tf.data.Dataset.zip((words_ds, midis_ds))
        if sequences is None:
            sequences = sample_ds.batch(MAX_SEQ_LENGTH + 1, drop_remainder=True)
        else:
            sequences = sequences.concatenate(sample_ds.batch(MAX_SEQ_LENGTH + 1, drop_remainder=True))

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    # max_midi_len = max([midi.size for midi in midis])
    # sequences = []
    # midis_by_sequence = []
    # for index, sample in enumerate(train_songs):
    #     sample_sequences = []
    #     for i in range(MAX_SEQ_LENGTH, len(sample)):
    #         sample_sequence = [x[0] for x in sample[i - MAX_SEQ_LENGTH:i + 1]]
    #         sample_sequences.append(sample_sequence)
    #         vector_sequence = [np.array(x[1]) for x in sample[i - MAX_SEQ_LENGTH:i + 1]]
    #         tmp = np.pad(vector_sequence[0], [(0, max_midi_len - (vector_sequence[0]).size)])
    #         for v in vector_sequence[1:]:
    #             tmp = np.add(tmp,v)
    #         midis_by_sequence.append(tmp)
    #     sequences.append(np.array(sample_sequences))
    # sequences = np.vstack(sequences)
    #
    # # divide the sequence into X and y
    # X = sequences[:, :-1]  # assign all but last words of a sequence to X
    # y = sequences[:, -1]  # assign last word of each sequence to
    # y = to_categorical(y, num_classes=VOCABULARY_SIZE)

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

    # midi_size = midis_by_sequence[0].size
    # model_wv = define_model(VOCABULARY_SIZE, EMBEDDING_SIZE, embedding_weights, midi_size)
    model_wv = CustomModel(VOCABULARY_SIZE, EMBEDDING_SIZE, embedding_weights)

    # compile network
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # from_logics=True since there is no
                                                                            # activation of the last dense
    model_wv.compile(loss=loss, optimizer='adam')
    model_wv.build(input_shape=[(MAX_SEQ_LENGTH, 1), (MAX_SEQ_LENGTH, 128)])

    # summarize defined model
    model_wv.summary()

    # fit network
    history = model_wv.fit(dataset, epochs=50, verbose=1)

    with open("run.dump", 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model_wv, word_tokeniser, max_midi_len


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

        # predict next word6
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

    [print(line) for line in text.split("&")]
