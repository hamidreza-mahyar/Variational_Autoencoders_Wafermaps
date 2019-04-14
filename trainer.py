from keras.models import Model, Sequential
from tqdm import tqdm_notebook, tqdm
from data_utils import batch_disc_real, batch_disc_fake, batch_gan
import numpy as np
import keras.backend as K
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import os, shutil
from functools import partial

CAT_DIM = 5
CONT_DIM = 2
NOISE_DIM = 64
BATCH_SIZE = 32
MA_SIZE = 100
TENSORBOARD_LOG_FILE = '/tmp/tensorflow'


def gaussian_loss(y_true, y_pred):
    mean = y_true[:, 0, :]
    log_stdev = y_true[:, 1, :]
    x = y_pred[:, 0, :]

    frac = K.square(x - mean) / (K.exp(log_stdev) + K.epsilon())
    return 0.5 * K.mean(np.log(2 * np.pi) + log_stdev + frac)

def disc_loss(y_true, y_pred):
    return 0.5 * (K.sum(y_true * K.square(y_pred - 1), axis=1) + K.sum((1-y_true) * K.square(y_pred), axis=1))

def gen_loss(y_true, y_pred):
    return 0.5 * K.sum(K.square(y_pred - 1), axis=1)

def zero_loss(y_true, y_pred):
    return y_true * 0

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

def make_trainable(net, val):
    net.trainable = val
    if any(isinstance(net, t) for t in [Model, Sequential]):
        for l in net.layers:
            make_trainable(l, val)

def write_log(callback, tags, logs, batch_no):
    values = []
    for tag, log in zip(tags, logs):
        values.append(tf.Summary.Value(tag=tag, simple_value=log))

    summary = tf.Summary(value=values)
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

def train(generator, discriminator, gan, train_data, nb_epoch=5000, loss={'d': [], 'g': []}):
    loss_iter = {'d': deque(maxlen=MA_SIZE), 'g': deque(maxlen=MA_SIZE)}
    lr_initial_gan = K.get_value(gan.optimizer.lr)
    lr_initial_disc = K.get_value(gan.optimizer.lr)

    if os.path.isdir(TENSORBOARD_LOG_FILE):
        shutil.rmtree(TENSORBOARD_LOG_FILE)
    callback = TensorBoard(
        log_dir=TENSORBOARD_LOG_FILE,
        histogram_freq=1
    )
    callback.set_model(gan)
    d_names = ['d_total', 'd_real', 'd_fake', 'd_cont', 'd_cat', 'd_lr']
    g_names = ['g_total', 'g_real', 'g_fake', 'g_cont', 'g_cat', 'g_lr']
    t = tqdm_notebook(range(nb_epoch))
    for e in t:
        if e%2 == 0:
            image_batch = train_data[np.random.randint(0, train_data.shape[0], size=BATCH_SIZE),:,:,:]
            #d_input, d_output = batch_disc_real(image_batch, BATCH_SIZE, CAT_DIM, CONT_DIM, noise=1-1*min(1, (10*e)/nb_epoch))
            d_input, d_output = batch_disc_real(image_batch, BATCH_SIZE, CAT_DIM, CONT_DIM, noise=0)
        else:
            d_input, d_output = batch_disc_fake(generator, BATCH_SIZE, CAT_DIM, CONT_DIM, NOISE_DIM, noise=0)

        d_loss = discriminator.train_on_batch(d_input, d_output)
        d_loss.append(K.get_value(discriminator.optimizer.lr))

        # Save discriminative loss
        if e%2 == 0:
            d_loss.insert(1, 0)
        else:
            d_loss.insert(2, 0)
        write_log(callback, d_names, d_loss, e)
        loss_iter['d'].append(d_loss)
        loss['d'].append(d_loss)

        gan_input, gan_output = batch_gan(BATCH_SIZE, CAT_DIM, CONT_DIM, NOISE_DIM)

        make_trainable(discriminator, False)
        g_loss = gan.train_on_batch(gan_input, gan_output)
        g_loss.append(K.get_value(gan.optimizer.lr))
        make_trainable(discriminator, True)

        # Save generative loss
        if e%2 == 0:
            g_loss.insert(1, 0)
        else:
            g_loss.insert(2, 0)
        write_log(callback, g_names, g_loss, e)
        loss_iter['g'].append(g_loss)
        loss['g'].append(g_loss)

        loss_avg = {
            'd': np.mean(loss_iter['d'], axis=0),
            'g': np.mean(loss_iter['g'], axis=0)
        }
        t.set_description("G: %.3f = %.3f + %.3f + %.3f + %.3f, lr = %.6f; D: %.3f = %.3f + %.3f + %.3f + %.3f, lr = %.6f" % (
            loss_avg['g'][0],
            loss_avg['g'][1],
            loss_avg['g'][2],
            loss_avg['g'][3],
            loss_avg['g'][4],
            loss_avg['g'][5],
            loss_avg['d'][0],
            loss_avg['d'][1],
            loss_avg['d'][2],
            loss_avg['d'][3],
            loss_avg['d'][4],
            loss_avg['d'][5]
        ))

        # Update learning rates
        lr_frac = 100 / (100+max(0, e-10000))
        K.set_value(gan.optimizer.lr, lr_frac * lr_initial_gan)
        K.set_value(discriminator.optimizer.lr, lr_frac * lr_initial_disc)
