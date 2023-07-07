from keras.models import load_model
import pickle
from Ex4 import TRAIN_VECTOR_PATH
import numpy as np
from keras.preprocessing.text import Tokenizer

def get_songs(path):
    with open(path, 'rb') as handle:
        t: dict = pickle.load(handle)
        return t


def generate_song():
    generator = load_model('generator.h5')
    decoder = load_model('decoder.h5')

    noise_dim = 16

    train, test = get_songs(TRAIN_VECTOR_PATH).values()
    mid_encoding = list()
    lyrics = list()
    for sample in train:
        sample_mid = [x[1] for x in sample]
        # noise = np.random.normal(0, 1, (noise_dim))
        # mid_encoding.append(np.concatenate((np.average(sample_mid, axis=0),noise)) )
        mid_encoding.append(np.average(sample_mid, axis=0))
        lyrics.append([x[0] for x in sample])

    tokenizer = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
    tokenizer.fit_on_texts(lyrics)
    index_word = tokenizer.index_word
    sequences = tokenizer.texts_to_sequences(lyrics)


    mid_encoding= mid_encoding[-5:]

    noise = np.random.normal(0, 1, (noise_dim))
    mid_noised_encoding = np.concatenate((np.array(mid_encoding), np.tile(noise, (5, 1))),
                                                       axis=1)

    encodings = generator.predict(mid_noised_encoding)

    songs = decoder.predict(encodings)

    song_tokens = np.argmax(songs, axis=-1)

    full_songs = list()
    for song in song_tokens:
        full_songs.append([index_word[token+1] for token in song])

    x=0

if __name__ == "__main__":
    generate_song()