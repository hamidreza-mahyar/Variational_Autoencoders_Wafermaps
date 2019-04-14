import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns


def plot_loss(losses):
    loss_g = np.array(losses['g']).T
    loss_d = np.array(losses['d']).T

    plt.figure(figsize=(10, 5))
    plt.plot(loss_g[0], label='generative loss')
    plt.plot(loss_d[0], label='discriminitive loss')
    plt.legend()
    plt.grid()


def plot_grid(generated_images, dim=(4,4), figsize=(10,10)):
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = generated_images[i,:,:,0]
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()


def plot_wafer(arr, out_file=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4), gridspec_kw = {'width_ratios': [1, 3]})

    ax1.imshow(arr)

    flat = arr.flatten()
    x = flat[~np.isnan(flat)]
    sns.distplot(x, ax=ax2)
    plt.ylabel('Frequency')
    plt.xlabel('Measurement value')
    if out_file == None:
        plt.show()
    else:
        plt.savefig(out_file)
