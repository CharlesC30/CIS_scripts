import numpy as np
import matplotlib.pyplot as plt

def plot_images_and_hist(*images, bins=1000):
    fig, axs = plt.subplots(len(images), 2)
    for im, ax in zip(images, axs):
        ax[0].imshow(im)
        ax[1].hist(im.ravel(), bins=bins)
    plt.show()