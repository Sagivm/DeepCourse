from Ex4.data.autoencoder import *
from Ex4.model.gan import build_generator_model, build_discriminator_model, build_gan_model
from keras.models import load_model, Functional


def main():
    # Build and import
    embedding_dim = 128
    noise_dim = 16
    mid_dim = 128

    # build models
    generator = build_generator_model(embedding_dim,mid_dim, noise_dim)
    discriminator = build_discriminator_model(embedding_dim, mid_dim)
    # gan = build_gan_model(generator, discriminator, (mid_dim + noise_dim,))
    autoencoder: Functional = load_model('autoencoder.h5')
    encoder = Model(autoencoder.input, autoencoder.layers[3].output)

    encoder.summary()
    # import songs
    #
    train,test = get_songs(TRAIN_VECTOR_PATH).values()
    mid_encoding = list()
    for sample in train:
        sample_mid = [x[1] for x in sample]
        noise = np.random.normal(0, 1, (noise_dim))
        mid_encoding.append(np.concatenate((np.average(sample_mid, axis=0),noise)) )

    x=0
    generator.predict(np.array(mid_encoding))
print("WTF")
main()
