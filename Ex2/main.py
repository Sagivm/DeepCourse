import tensorflow as tf
from Ex2.util.format_dataset import format_dataset
from Ex2 import TRAIN_PATH, TEST_PATH , BATCH_SIZE, EPOCHS
from Ex2.siamese_model import build_base_model, build_siamese_model, batch_generator
from keras.optimizers import SGD,Adam
import numpy as np
import imageio

def get_dataset():
    train = format_dataset(TRAIN_PATH)[::5]
    test = format_dataset(TEST_PATH)[::5]
    model = build_siamese_model((250, 250, 1))
    train_batch_generator = batch_generator(train, BATCH_SIZE)
    val_batch_generator = batch_generator(test, BATCH_SIZE)

    opt = SGD(lr=0.001, momentum=0.5)
    # opt = Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    model.fit(
            train_batch_generator,
            steps_per_epoch=len(train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_batch_generator,
            validation_steps=len(test) // BATCH_SIZE)


get_dataset()
