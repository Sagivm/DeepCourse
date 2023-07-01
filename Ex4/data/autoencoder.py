from gensim.models import KeyedVectors, LdaModel

from keras.layers import Input, Dense, Embedding, LSTM, Activation
from keras.models import Model,Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam
import pickle
import numpy as np
from tensorflow.python.autograph.operators.py_builtins import max_

from Ex4 import TRAIN_VECTOR_PATH, LMODEL_PATH
from keras.preprocessing.text import Tokenizer
from keras.layers import RepeatVector,TimeDistributed


def get_songs(path):
    with open(path, 'rb') as handle:
        t: dict = pickle.load(handle)
        return t


def build_autoencoder():

    #train,test = get_songs(TRAIN_VECTOR_PATH).values()
    train_text = get_songs('data/tokenized_train_text.pkl')
    test_text = get_songs('data/tokenized_train_text.pkl')
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
    embedding_layer = Embedding(input_dim=len(word_index)+1,output_dim=EMBEDDING_DIM,weights=[embedding_weights], trainable=False)(input_layer)


    # Define the encoding layer
    encoder = LSTM(128, return_sequences=False)(embedding_layer)

    encoder = Dense(128,activation='relu')(encoder)
    # # Repeat the encoded representation
    # repeat_layer = RepeatVector(max_sequence_length)(encoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    # Define the decoding layer
    # decoder = Input((128,))
    decoder = Dense(128,activation='relu')(encoder)
    decoder = RepeatVector(max_sequence_length)(decoder)
    decoder = LSTM(128, return_sequences=True,input_shape=(1,64))(decoder)
    decoder = Dense(len(word_index)+1,activation='softmax')(decoder)
    # logits = TimeDistributed(Dense(len(word_index)+1))(decoder)

    # Create the autoencoder model
    # autoencoder_outputs = decoder(encoder_model(input_layer))
    # autoencoder = Model(input=input_layer, outputs=autoencoder_outputs)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    # autoencoder.compile(loss='sparse_categorical_crossentropy',
    #           optimizer=Adam(1e-3),
    #           metrics=['accuracy'])
    # autoencoder.summary()
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Print the model summary
    autoencoder.summary()

    # Train the autoencoder
    autoencoder.fit(padded_sequences, padded_sequences, epochs=50, batch_size=16)

    autoencoder.save('autoencoder.h5')

# build_autoencoder()