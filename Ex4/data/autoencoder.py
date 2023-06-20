from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam
import pickle
import numpy as np
from tensorflow.python.autograph.operators.py_builtins import max_

from Ex4 import TRAIN_VECTOR_PATH
from keras.preprocessing.text import Tokenizer
from keras.layers import RepeatVector,TimeDistributed


def get_songs(path):
    with open(path, 'rb') as handle:
        t: dict = pickle.load(handle)
        return t


def autoencoder():

    train,test = get_songs(TRAIN_VECTOR_PATH).values()
    train_text = get_songs('data/tokenized_train_text.pkl')
    test_text = get_songs('data/tokenized_train_text.pkl')
    #print(str(data)[:100])
    # train = data['train']
    # Define the maximum number of words to consider in the vocabulary
    tokenizer = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
    tokenizer.fit_on_texts(train_text+test_text)
    max_words = len(tokenizer.word_index) + 1

    # Define the maximum length of input sequences
    max_sequence_length = max([len(x.split(" ")) for x in train_text + test_text])

    # Define the embedding dimension
    embedding_dim = 100

    # Load and preprocess the text data
    print("Train autoencoder")

    sequences = tokenizer.texts_to_sequences(train_text)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Define the input shape
    encoder_inputs = Input(shape=(max_sequence_length,), name='Encoder-Input')
    emb_layer = Embedding(max_words, embedding_dim, input_length=max_sequence_length, name='Body-Word-Embedding')
    x = emb_layer(encoder_inputs)
    state_h = LSTM(8, activation='relu', name='Encoder-Last-LSTM')(x)
    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
    seq2seq_encoder_out = encoder_model(encoder_inputs)

    decoded = RepeatVector(max_sequence_length)(seq2seq_encoder_out)
    decoder_lstm = LSTM(8, return_sequences=True, name='Decoder-LSTM-before')
    decoder_lstm_output = decoder_lstm(decoded)
    decoder_dense = Dense(max_words, activation='softmax', name='Final-Output-Dense-before')
    decoder_outputs = decoder_dense(decoder_lstm_output)

    # Compile the autoencoder model
    seq2seq_Model = Model(encoder_inputs, decoder_outputs)
    ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
                                 clipnorm=1.)
    seq2seq_Model.compile(optimizer=ADAM, loss='mse')
    seq2seq_Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    seq2seq_Model.summary()
    history = seq2seq_Model.fit(padded_sequences, np.expand_dims(padded_sequences, -1),
                                batch_size=16,
                                epochs=20)

    # return encoder

autoencoder()