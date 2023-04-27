import tensorflow as tf
from Ex2.util.format_dataset import format_dataset
from Ex2 import TRAIN_PATH, TEST_PATH, BATCH_SIZE, EPOCHS
from Ex2.siamese_model import build_base_model, build_siamese_model, batch_generator
# import Ex2.siamese_model
from tensorflow.keras.optimizers import SGD
# import numpy as np
# import imageio
import random


def get_dataset():
    # Acquiring the datasets abd divide them into train, validation and test
    train = format_dataset(TRAIN_PATH)
    random.shuffle(train)
    train = train[:-200]
    val = train[-200:]
    test = format_dataset(TEST_PATH)

    # Building the model
    model = build_siamese_model((250, 250, 1))

    # Building generators for all sets

    train_batch_generator = batch_generator(train, BATCH_SIZE)
    val_batch_generator = batch_generator(val, BATCH_SIZE)
    test_generator = batch_generator(test, BATCH_SIZE, False)
    return (train_batch_generator, len(train)), (val_batch_generator, len(val)), test_generator


def train_model(train, val):
    train_batch_generator, train_count = train
    val_batch_generator, val_count = val

    model = build_siamese_model((250, 250, 1))

    # Model compilation additions
    opt = SGD(learning_rate=0.001, momentum=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.summary()

    # Training the model
    history = model.fit(
        train_batch_generator,
        steps_per_epoch=train_count // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_batch_generator,
        validation_steps=val_count // BATCH_SIZE,
        callbacks=callback)
    return model, history


if __name__ == '__main__':
    train, val, test = get_dataset()
    model, history = train_model(train, val)
    model.evaluate(test)

