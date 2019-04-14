import numpy as np
from keras.datasets import mnist
import os
from PIL import Image
from tqdm import tqdm_notebook

def repeat_dims(a, axis):
    a = np.expand_dims(a, axis=axis)
    return np.repeat(a, 2, axis=axis)


def sample_noise(noise_scale, batch_size, noise_dim):
    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))


def sample_cat(batch_size, cat_dim):
    y = np.zeros((batch_size, cat_dim), dtype='float32')
    random_y = np.random.randint(0, cat_dim, size=batch_size)
    y[np.arange(batch_size), random_y] = 1

    return y


def batch_gan(batch_size, cat_dim, cont_dim, noise_dim, noise_scale=0.5):
    X_noise = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.ones((X_noise.shape[0], 1), dtype=np.uint8)

    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)

    # Repeat y_cont to accomodate for keras' loss function conventions
    y_cont_target = repeat_dims(y_cont, 1)

    return [y_cat, y_cont, X_noise], [y_gen, y_cat, y_cont_target]


def batch_disc_real(X_real_batch, batch_size, cat_dim, cont_dim, noise_scale=0.5, label_smoothing=True, label_flipping=0, noise=0.9):
    idx = np.random.randint(X_real_batch.shape[0], size=batch_size)
    X_disc = X_real_batch[idx,:]
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)
    if label_smoothing:
        y_disc = np.random.uniform(low=0.9, high=1, size=X_disc.shape[0])
    else:
        y_disc = np.ones(X_disc.shape[0])

    if label_flipping > 0:
        p = np.random.binomial(1, label_flipping)
        if p > 0:
            y_disc = 1 - y_disc

    if noise > 0:
        X_disc = X_disc + noise * np.random.normal(scale=0.5, size=X_disc.shape)

    # Repeat y_cont to accomodate for keras' loss function conventions
    y_cont = repeat_dims(y_cont, 1)

    return X_disc, [y_disc, y_cat, y_cont]


def batch_disc_fake(generator, batch_size, cat_dim, cont_dim, noise_dim, noise_scale=0.5, label_flipping=0, noise=0.9):
    # Pass noise to the generator
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)

    # Produce an output
    X_disc = generator.predict([y_cat, y_cont, noise_input], batch_size=batch_size)
    y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)

    if label_flipping > 0:
        p = np.random.binomial(1, label_flipping)
        if p > 0:
            y_disc = 1 - y_disc

    if noise > 0:
        X_disc = X_disc + noise * np.random.normal(scale=0.5, size=X_disc.shape)

    # Repeat y_cont to accomodate for keras' loss function conventions
    y_cont = repeat_dims(y_cont, 1)

    return X_disc, [y_disc, y_cat, y_cont]


def batch_disc(image_batch, generator, batch_size, cat_dim, cont_dim, noise_dim):
    d_input_real, d_output_real = batch_disc_real(image_batch, batch_size, cat_dim, cont_dim)
    d_input_fake, d_output_fake = batch_disc_fake(generator, batch_size, cat_dim, cont_dim, noise_dim)

    return np.concatenate([d_input_real, d_input_fake]), [
        np.concatenate([d_output_real[0], d_output_fake[0]]),
        np.concatenate([d_output_real[1], d_output_fake[1]]),
        np.concatenate([d_output_real[2], d_output_fake[2]])
    ]


def load_mnist():
    img_rows, img_cols = 28, 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 128 - 1
    X_test = X_test / 128 - 1

    #y_train[y_train > 0] = 0
    #y_train[y_train == 0] = 1
    #print(y_train)

    #X_train = np.repeat(np.expand_dims(X_train[0], 0), 1000, axis=0)

    return X_train


def load_mnist_big():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    images = []
    for img in tqdm_notebook(X_train):
        img = img.reshape(28,28)
        img = Image.fromarray(np.uint8(img))
        img = img.resize((64, 64), Image.ANTIALIAS)
        images.append(np.array(img))

    X_train = np.array(images)
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
    X_train = X_train.astype('float32')
    X_train = X_train / 128 - 1

    return X_train

def load_wafers(file):
    img_rows, img_cols = 193, 115
    file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + file + '.npy'))
    data = np.load(file)
    print(data.shape)
    #assert data.shape[1] == img_rows and data.shape[2] == img_cols

    images = []
    for img in tqdm_notebook(data):
        img = Image.fromarray(np.uint8((img + 1) * 128))
        img = img.resize((128, 128), Image.ANTIALIAS)
        images.append(np.array(img))

    data = np.array(images)
    data = data.reshape(data.shape[0], 128, 128, 1)
    #data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    data = data.astype('float32')
    data = data / 128 - 1

    return data
