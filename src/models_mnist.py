import keras.backend as K
from keras.regularizers import *
from keras.layers.normalization import *
from keras.layers import Input, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten, Lambda
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU


def generator_model(cat_dim, cont_dim, noise_dim):
    H = Sequential()

    H.add(Dense(1024, input_shape=(cat_dim + cont_dim + noise_dim,)))
    H.add(BatchNormalization())
    H.add(Activation('relu'))

    H.add(Dense(128*7*7))
    H.add(BatchNormalization())
    H.add(Activation('relu'))
    H.add(Reshape([7, 7, 128]))

    H.add(UpSampling2D(size=(2, 2)))
    H.add(Conv2D(128, (3, 3), padding='same'))
    H.add(BatchNormalization())
    H.add(Activation('relu'))

    H.add(UpSampling2D(size=(2, 2)))
    H.add(Conv2D(64, (3, 3), padding='same'))
    H.add(BatchNormalization())
    H.add(Activation('relu'))

    H.add(Conv2D(1, (3, 3), padding='same'))
    H.add(Activation('tanh', name='out_image'))

    gen_in = [
        Input(shape=(cat_dim,), name='in_cat'),
        Input(shape=(cont_dim,), name='in_cont'),
        Input(shape=(noise_dim,), name='in_noise')
    ]
    gen_out = H(concatenate(gen_in))

    return Model(gen_in, gen_out)


def classifier(discriminator, cat_dim, cont_dim):
    H = Dense(128)(discriminator)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)

    H_Y = Dense(cat_dim, activation='softmax', name='out_cat')(H)

    def linmax(x):
        return K.maximum(x, -16)

    def linmax_shape(input_shape):
        return input_shape

    H_C_mean = Dense(cont_dim, activation='linear')(H)
    H_C_logstd = Dense(cont_dim)(H)
    H_C_logstd = Lambda(linmax, output_shape=linmax_shape)(H_C_logstd)
    # Reshape Q to nbatch, 1, cont_dim[0]
    H_C_mean = Reshape((1, cont_dim), name='out_cont_mean')(H_C_mean)
    H_C_logstd = Reshape((1, cont_dim), name='out_cont_logstd')(H_C_logstd)
    H_C = concatenate([H_C_mean, H_C_logstd], axis=1)

    return H_Y, H_C


def discriminator_model(cat_dim, cont_dim, dropout_rate = 0.25):
    H = Sequential()

    H.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(28,28,1)))
    H.add(LeakyReLU(0.2))
    #H.add(Dropout(dropout_rate))

    H.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    H.add(BatchNormalization())
    H.add(LeakyReLU(0.2))
    #H.add(Dropout(dropout_rate))
    H.add(Flatten())

    H.add(Dense(1024))
    H.add(BatchNormalization())
    H.add(LeakyReLU(0.2))
    #H.add(Dropout(dropout_rate))

    disc_in = Input(shape=(28, 28, 1), name='in_image')
    features = H(disc_in)
    disc_out = Dense(1, activation='sigmoid', name='out_binary')(features)

    Q_Y, Q_C = classifier(features, cat_dim, cont_dim)

    return Model(disc_in, [disc_out, Q_Y, Q_C])


def gan_model(generator, discriminator, cat_dim, cont_dim, noise_dim):
    gan_in = [
        Input(shape=(cat_dim,), name='in_cat'),
        Input(shape=(cont_dim,), name='in_cont'),
        Input(shape=(noise_dim,), name='in_noise')
    ]

    H = generator(gan_in)
    gan_out = discriminator(H)

    return Model(gan_in, gan_out)