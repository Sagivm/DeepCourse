from Ex4.data.autoencoder import *
from Ex4.model.gan import build_generator_model, build_discriminator_model, build_gan_model
from keras.models import load_model, Functional
import random
import tensorflow as tf
from keras.losses import BinaryCrossentropy, mean_squared_error



def main():
    # Build and import
    embedding_dim = 128
    noise_dim = 16
    mid_dim = 128

    # build models
    generator = build_generator_model(embedding_dim, mid_dim, noise_dim)
    discriminator = build_discriminator_model(embedding_dim, mid_dim)
    # gan = build_gan_model(generator, discriminator, (mid_dim + noise_dim,))
    autoencoder: Functional = load_model('autoencoder.h5')
    encoder = Model(autoencoder.input, autoencoder.layers[3].output)

    encoder.summary()
    # import songs
    #
    lyrics = list()
    train, test = get_songs(TRAIN_VECTOR_PATH).values()
    train = train[:595]
    mid_encoding = list()
    for sample in train:
        sample_mid = [x[1] for x in sample]
        # noise = np.random.normal(0, 1, (noise_dim))
        # mid_encoding.append(np.concatenate((np.average(sample_mid, axis=0),noise)) )
        mid_encoding.append(np.average(sample_mid, axis=0))
        lyrics.append([x[0] for x in sample])
    epochs = 50
    batch_size = 35
    for epoch in range(epochs):
        for batch in range(595//batch_size):
            noise = np.random.normal(0, 1, (noise_dim))
            song_batch_lyrics = lyrics[batch * batch_size:(batch + 1) * batch_size]
            # Transfer to autoencoder

            tokenizer = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
            tokenizer.fit_on_texts(song_batch_lyrics)
            word_index = tokenizer.word_index
            sequences = tokenizer.texts_to_sequences(song_batch_lyrics)

            # Pad sequences to a fixed length
            max_sequence_length = 1521
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

            song_encodings = encoder.predict(padded_sequences)

            # Transfer to generator
            mid_batch_encoding = mid_encoding[batch * batch_size:(batch + 1) * batch_size]
            # mid_encoding.append(np.concatenate((np.average(sample_mid, axis=0), noise)))
            mid_batch_noised_encoding = np.concatenate((np.array(mid_batch_encoding), np.tile(noise, (batch_size, 1))),
                                                       axis=1)
            generated_samples = generator.predict(mid_batch_noised_encoding)

            # Shuffle
            shuffled = list()
            for x, y, z in list(zip(generated_samples, song_encodings, mid_batch_encoding)):
                shuffled.append((x, y, z, 1)) if random.random() < 0.5 else shuffled.append((y, x, z, 0))

            x = [t[0] for t in shuffled]
            y = [t[1] for t in shuffled]
            z = [t[2] for t in shuffled]
            true_labels = [t[3] for t in shuffled]
            false_labels = [1 if t==0 else 0 for t in shuffled]

            # enter to the descriminator
            # predictions = discriminator.predict([np.array(x), np.array(y), np.array(z)])

            discriminator_loss = discriminator.train_on_batch([np.array(x), np.array(y), np.array(z)], np.array(true_labels))

            # discriminator.trainable = False
            # combined = Sequential(generator, discriminator)
            # combined.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # combined.train_on_batch([np.array(x), np.array(y), np.array(z)], np.array(true_labels))

            with tf.GradientTape() as tape:
                predictions = discriminator([np.array(x), np.array(y), np.array(z)])
                g_loss = mean_squared_error(true_labels, predictions)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(grads,generator.trainable_variables))
            # with tf.GradientTape() as tape:
            #     pass
            # tape.gradient(discriminator_loss,generator.trainable_variables)
            # optimizer = generator.optimizer

            x=0
            # AM.train_on_batch([np.array(x), np.array(y), np.array(z)], np.array(true_label))


print("WTF")
main()
