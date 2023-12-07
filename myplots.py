import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches

def plot_images_and_hist(*images, bins=1000):
    fig, axs = plt.subplots(len(images), 2)
    for im, ax in zip(images, axs):
        ax[0].imshow(im)
        ax[1].hist(im.ravel(), bins=bins)
    plt.show()

def draw_bbox(bbox, ax, color, **kwargs):
    min_y, min_x, max_y, max_x = bbox
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle((min_x - 0.5, min_y - 0.5), width, height, edgecolor=color, facecolor="none", **kwargs)
    ax.add_patch(rect)


def y_profile(image, y, image_ax, profile_ax, line_color="red"):
    profile_vals = image[y, :]
    image_ax.imshow(image)
    image_ax.axhline(y=y, color=line_color)
    image_ax.set_xticks([]), image_ax.set_yticks([])
    profile_ax.plot(profile_vals)
    return profile_vals


