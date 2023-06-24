from keras.layers import Input, Dense, Embedding, LSTM, Activation
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

    #train,test = get_songs(TRAIN_VECTOR_PATH).values()
    train_text = get_songs('data/tokenized_train_text.pkl')
    test_text = get_songs('data/tokenized_train_text.pkl')
    #print(str(data)[:100])
    # train = data['train']
    # Define the maximum number of words to consider in the vocabulary
    tokenizer = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
    tokenizer.fit_on_texts(train_text+test_text)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(train_text)

    # Pad sequences to a fixed length
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Define the input shape
    input_shape = (max_sequence_length,)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Add an embedding layer
    embedding_dim = 32
    embedding_layer = Embedding(input_dim=len(word_index)+1,output_dim=embedding_dim)(input_layer)


    # Define the encoding layer
    encoder = LSTM(64, return_sequences=False)(embedding_layer)

    # Repeat the encoded representation
    repeat_layer = RepeatVector(max_sequence_length)(encoder)

    # Define the decoding layer
    decoder = LSTM(64, return_sequences=True)(repeat_layer)

    logits = TimeDistributed(Dense(len(word_index)+1))(decoder)

    # Create the autoencoder model
    autoencoder = Model(input_layer, Activation('softmax')(logits))
    autoencoder.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(1e-3),
              metrics=['accuracy'])
    autoencoder.summary()
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Print the model summary
    autoencoder.summary()

    # Train the autoencoder
    autoencoder.fit(padded_sequences, padded_sequences, epochs=50, batch_size=32)

    # max_words = len(tokenizer.word_index) + 1
    #
    # # Define the maximum length of input sequences
    # max_sequence_length = max([len(x.split(" ")) for x in train_text + test_text])
    #
    # # Define the embedding dimension
    # embedding_dim = 64
    # encoding_dim = 128
    # sequences = tokenizer.texts_to_sequences(train_text)
    # padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length,padding='post')
    #
    # # Load and preprocess the text data
    # print("Train autoencoder")
    #
    # # Define the input shape
    # input_shape = (max_sequence_length,)  # Replace max_sequence_length with the maximum length of your text sequences
    #
    # # Define the encoding dimension
    # encoding_dim = 64  # Choose an appropriate dimension for the compressed representation
    #
    # # Define the input layer
    # input_layer = Input(shape=input_shape)
    #
    # # Define the encoding layer
    # encoder = Embedding(max_words, embedding_dim)(input_layer)
    # encoder = LSTM(encoding_dim)(encoder)
    #
    # # Repeat the encoded representation
    # repeat_layer = RepeatVector(max_sequence_length)(encoder)
    #
    # # Define the decoding layer
    # decoder = LSTM(input_shape[0], return_sequences=True)(repeat_layer)
    #
    # # Create the autoencoder model
    # autoencoder = Model(input_layer, decoder)
    #
    # # Compile the model
    # autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    #
    # # Print the model summary
    # autoencoder.summary()
    #
    # autoencoder.fit(padded_sequences, padded_sequences, epochs=100)

    # inputs = Input(shape=(max_sequence_length,))
    # encoder1 = Embedding(max_words, 128)(inputs)
    # encoder2 = LSTM(128)(encoder1)
    # encoder3 = RepeatVector(10)(encoder2)
    # # decoder output model
    # decoder1 = LSTM(128, return_sequences=True)(encoder3)
    # outputs = TimeDistributed(Dense(max_words, activation='softmax'))(decoder1)
    # # tie it together
    # model = Model(inputs=inputs, outputs=outputs)
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.fit(padded_sequences, padded_sequences, epochs=100)
    #
    #
    # input_song = Input(shape=(max_sequence_length,))
    # embedding = Embedding(max_words, embedding_dim)(input_song)
    # encoded = LSTM(encoding_dim)(embedding)
    #
    # # decoded = RepeatVector(10)(encoded)
    # decoded = LSTM(max_sequence_length, return_sequences=True)(encoded)
    #
    # # Compile and train the autoencoder
    # autoencoder = Model(input_song, decoded)
    # autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.fit(padded_sequences, padded_sequences, epochs=100)

    #
    # autoencoder = Model(input_layer, decoder)
    # # Define the input shape
    # encoder_inputs = Input(shape=(max_sequence_length,), name='Encoder-Input')
    # emb_layer = Embedding(max_words, embedding_dim, input_length=max_sequence_length, name='Body-Word-Embedding',)
    # x = emb_layer(encoder_inputs)
    # state_h = LSTM(64, activation='relu', name='Encoder-Last-LSTM')(x)
    # encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
    # seq2seq_encoder_out = encoder_model(encoder_inputs)
    #
    # decoded = RepeatVector(max_sequence_length)(seq2seq_encoder_out)
    # decoder_lstm = LSTM(64, return_sequences=True, name='Decoder-LSTM-before')
    # decoder_lstm_output = decoder_lstm(decoded)
    # decoder_dense = Dense(max_words, activation='softmax', name='Final-Output-Dense-before')
    # decoder_outputs = decoder_dense(decoder_lstm_output)
    #
    # # Compile the autoencoder model
    # seq2seq_Model = Model(encoder_inputs, decoder_outputs)
    # ADAM = Adam(lr=0.0001)
    # #seq2seq_Model.compile(optimizer=ADAM, loss='sparse_categorical_crossentropy')
    # seq2seq_Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # seq2seq_Model.summary()
    # history = seq2seq_Model.fit(padded_sequences, np.expand_dims(padded_sequences, -1),
    #                             batch_size=32,
    #                             epochs=40)

    # return encoder

autoencoder()