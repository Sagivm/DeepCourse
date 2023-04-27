import tensorflow as tf
from Ex2.util.format_dataset import format_dataset
from Ex2 import TRAIN_PATH, TEST_PATH , BATCH_SIZE, EPOCHS
from Ex2.siamese_model import build_base_model, build_siamese_model, batch_generator,test_batch_generator
import Ex2.siamese_model
from keras.optimizers import SGD,Adam
import numpy as np
import imageio
import random

def get_dataset():
    train = format_dataset(TRAIN_PATH)[::2]
    random.shuffle(train)

    train = train[:-100]
    val = train[-100:]

    test = format_dataset(TEST_PATH)[::8]
    # random.shuffle(test)
    # test = test[::4]
    model = build_siamese_model((250, 250, 1))
    train_batch_generator = batch_generator(train, BATCH_SIZE)
    val_batch_generator = batch_generator(val, BATCH_SIZE)
    test_batch_generator = Ex2.siamese_model.test_batch_generator(test, BATCH_SIZE)

    opt = SGD(lr=0.001, momentum=0.5)
    # opt = Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.fit(
            train_batch_generator,
            steps_per_epoch=len(train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_batch_generator,
            validation_steps=len(val) // BATCH_SIZE,
            callbacks=callback)

    model.evaluate(test_batch_generator)
    x=0

get_dataset()
