import pickle

import pretty_midi
import pandas as pd
import numpy as np

from Ex3 import TRAIN_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH, TEST_PATH
from Ex3.data.pre_process import read_songs


def notes_durations(midi: pretty_midi.PrettyMIDI):
    notes = [None] * 128  # 128 optional instruments
    for ins in midi.instruments:
        if notes[ins.program] is not None:
            continue
        notes[ins.program] = np.array([[note.pitch, note.start, note.end] for note in ins.notes])
    return notes


def get_pitch(timestamps: np.ndarray, ts):
    if timestamps is None:
        return 0

    mask: np.ndarray = np.logical_and(timestamps[:, 1] <= ts, ts <= timestamps[:, 2])
    idx = np.argmax(mask)
    if idx == 0:
        if not mask[0]:
            return 0
    return timestamps[idx, 0]


def create_vectors(song: str, midi: pretty_midi.PrettyMIDI):
    notes = notes_durations(midi)

    return [(word, np.array([get_pitch(instrument, beat) for instrument in notes])) for beat, word in
            zip(midi.get_beats(), song)]


def build_set(s):
    vectors = []
    for sample, midi_path in zip(*s):
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except:
            vectors.append([(word, [0] * 128) for word in sample])
        else:
            vectors.append(create_vectors(sample, midi))
    return vectors[:20]


if __name__ == '__main__':
    train, test = read_songs(TRAIN_PATH, TEST_PATH, TRAIN_VECTOR_PATH, TSET_VECTOR_PATH)

    with open('lyrics+pitches.pkl', 'wb') as f:
        pickle.dump({'train': build_set(train), 'test': build_set(test)}, f, protocol=pickle.HIGHEST_PROTOCOL)
