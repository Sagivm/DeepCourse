from Ex4.data.autoencoder import *
from Ex4.model.gan import build_generator_model, build_discriminator_model
from keras.models import load_model, Functional
import random
import tensorflow as tf
from keras.losses import BinaryCrossentropy, mean_squared_error
from Ex4 import TRAIN_VECTOR_PATH


def main():
    # Build and import
    embedding_dim = 256
    noise_dim = 16
    mid_dim = 128

    # build and get models
    generator = build_generator_model(embedding_dim, mid_dim, noise_dim)
    discriminator = build_discriminator_model(embedding_dim, mid_dim)
    encoder: Functional = load_model('encoder.h5')

    generator_optimizer = generator.optimizer
    discriminator_optimizer = discriminator.optimizer

    encoder.summary()
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
    epochs = 64
    batch_size = 35
    for epoch in range(epochs):
        for batch in range(595 // batch_size):
            noise = np.random.normal(0, 1, (noise_dim))
            song_batch_lyrics = lyrics[batch * batch_size:(batch + 1) * batch_size]
            # Create song encoding to be passed to the autoencoder

            tokenizer = Tokenizer(filters='!"#$%()*+,./:;<=>?@[\\]^_{|}~\t\n', )
            tokenizer.fit_on_texts(song_batch_lyrics)
            word_index = tokenizer.word_index
            sequences = tokenizer.texts_to_sequences(song_batch_lyrics)

            # Pad sequences to a fixed length
            max_sequence_length = 1521
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

            # get batch songs encodings
            song_encodings = encoder.predict(padded_sequences)

            # Append noise to mid encodings
            mid_batch_encoding = mid_encoding[batch * batch_size:(batch + 1) * batch_size]
            mid_batch_noised_encoding = np.concatenate((np.array(mid_batch_encoding), np.tile(noise, (batch_size, 1))),
                                                       axis=1)
            # Function defenition for manual loss calculation
            cross_entropy = BinaryCrossentropy(from_logits=True)

            with tf.GradientTape() as gen_tape:

                # Get generator samples
                generated_samples = generator(mid_batch_noised_encoding, training=True)

                # shuffle samples internal order to fool discriminator
                shuffled = list()
                for x, y, z in list(zip(generated_samples, song_encodings, mid_batch_encoding)):
                    shuffled.append((x, y, z, 1)) if random.random() < 0.5 else shuffled.append((y, x, z, 0))

                x = [t[0] for t in shuffled]
                y = [t[1] for t in shuffled]
                z = [t[2] for t in shuffled]
                true_labels = [t[3] for t in shuffled]
                false_labels = [1 if t == 0 else 0 for t in shuffled]

                # Compute discriminator's loss and gradients manually
                with tf.GradientTape() as disc_tape:
                    discriminator_output = discriminator([np.array(x), np.array(y), np.array(z)])
                    discriminator_loss = cross_entropy(np.array(true_labels).reshape(-1, 1), discriminator_output)
                gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, discriminator.trainable_variables))
            # Compute generator's loss and gradients manually
            with tf.GradientTape() as gen_tape:
                generated_data = generator(mid_batch_noised_encoding, training=True)
                fake_output = discriminator([generated_data, song_encodings, np.array(z)])
                generator_loss = cross_entropy(np.array(false_labels).reshape(-1, 1), fake_output)

            gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            print(
                f"Epoch {epoch + 1}/{epochs}, Step {batch + 1}/{595 // batch_size}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

    # Save trained generator and discriminator
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')

main()
