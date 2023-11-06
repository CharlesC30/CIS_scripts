import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches

def plot_images_and_hist(*images, bins=1000):
    fig, axs = plt.subplots(len(images), 2)
    for im, ax in zip(images, axs):
        ax[0].imshow(im)
        ax[1].hist(im.ravel(), bins=bins)
    plt.show()

def draw_bbox(bbox, ax, color):
    min_y, min_x, max_y, max_x = bbox
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle((min_x - 0.5, min_y - 0.5), width, height, edgecolor=color, facecolor="none", linewidth=2)
    ax.add_patch(rect)

