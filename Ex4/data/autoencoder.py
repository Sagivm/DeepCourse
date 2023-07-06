import pickle

import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Input, Dense, Embedding, LSTM, Activation, Bidirectional
from keras.layers import RepeatVector, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from Ex4 import LMODEL_PATH


def get_songs(path):
    with open(path, 'rb') as handle:
        t: dict = pickle.load(handle)
        return t


def build_autoencoder():

    #train,test = get_songs(TRAIN_VECTOR_PATH).values()
    train_text = get_songs('data/tokenized_train_text.pkl')
    test_text = get_songs('data/tokenized_test_text.pkl')
    tokenizer = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
    tokenizer.fit_on_texts(train_text+test_text)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(train_text)

    # Pad sequences to a fixed length
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    VOCABULARY_SIZE = len(word_index) + 1
    EMBEDDING_DIM = 300
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_DIM))
    word2vec = KeyedVectors.load_word2vec_format(LMODEL_PATH, binary=True)
    # copy vectors from word2vec model to the words present in corpus
    for word, index in tokenizer.word_index.items():
        try:
            embedding_weights[index, :] = word2vec[word]
        except KeyError:
            pass

    # Define the input shape
    input_shape = (max_sequence_length,)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Add an embedding layer
    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=EMBEDDING_DIM,
                                input_length=max_sequence_length,
                                weights=[embedding_weights], trainable=False,
                                name='Embedding')(input_layer)

    # Define the encoding layer
    encoder_out = Bidirectional(LSTM(128, return_sequences=False, ))(embedding_layer)
    encoder_model = Model(inputs=input_layer, outputs=encoder_out, name='EncoderModel')

    encoder = encoder_model(input_layer)
    # Repeat the encoded representation
    repeat_layer = RepeatVector(max_sequence_length)(encoder)

    # Define the decoding layer
    decoder = Bidirectional(LSTM(128, return_sequences=True))(repeat_layer)

    logits = TimeDistributed(Dense(len(word_index) + 1))(decoder)

    # Create the autoencoder model
    autoencoder = Model(input_layer, Activation('softmax')(logits), name='Autoencoder')

    # Print the model summary
    autoencoder.summary()

    # Compile the model
    autoencoder.compile(loss='sparse_categorical_crossentropy',
                        optimizer=Adam(1e-3))
    # Compile the model
    # autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # autoencoder.summary()

    # Train the autoencoder
    autoencoder.fit(padded_sequences, np.expand_dims(padded_sequences, -1),
                    epochs=20, batch_size=32)

    encoder_model.save('encoder.h5')
    autoencoder.save('autoencoder.h5')

# build_autoencoder()