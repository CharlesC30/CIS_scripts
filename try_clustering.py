import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import SpectralClustering


def pad_images(images, target_size):
    """
    Pad a collection of images with zeros to a specified target size.

    Parameters:
    - images: List of 2D NumPy arrays representing images.
    - target_size: Tuple (target_height, target_width) representing the desired size.

    Returns:
    - List of padded images.
    """
    padded_images = []
    target_height, target_width = target_size

    for image in images:
        # Get the current image dimensions
        current_height, current_width = image.shape

        # Calculate the amount of padding needed
        pad_height = max(0, target_height - current_height)
        pad_width = max(0, target_width - current_width)

        # Pad the image with zeros
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        padded_images.append(padded_image)

    return padded_images


def pad_images_center(images, target_size):
    """
    Pad a collection of images with zeros to a specified target size,
    keeping the original image centered.

    Parameters:
    - images: List of 2D NumPy arrays representing images.
    - target_size: Tuple (target_height, target_width) representing the desired size.

    Returns:
    - List of padded images.
    """
    padded_images = []
    target_height, target_width = target_size

    for image in images:
        # Get the current image dimensions
        current_height, current_width = image.shape

        # Calculate the amount of padding needed on each side
        pad_height_top = max(0, (target_height - current_height) // 2)
        pad_height_bottom = max(0, target_height - current_height - pad_height_top)
        pad_width_left = max(0, (target_width - current_width) // 2)
        pad_width_right = max(0, target_width - current_width - pad_width_left)

        # Pad the image with zeros while keeping it centered
        padded_image = np.pad(
            image,
            ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
            mode='constant',
            constant_values=0
        )
        padded_images.append(padded_image)

    return padded_images

if __name__ == "__main__":
    pore_candidates = [np.load(f"pore_candidates/{filename}") for filename in os.listdir("pore_candidates")]
    pore_candidates = [pc for pc in pore_candidates if (pc.shape[0] < 50 and pc.shape[1] < 50)]
    padded_pore_candidates = pad_images_center(pore_candidates, (50, 50))
    # for p in padded_pore_candidates:
    #     plt.imshow(p)
    #     plt.show()
    print(len(padded_pore_candidates))
    flattened_candidates = [np.ravel(pc) for pc in padded_pore_candidates]
    data = np.array(flattened_candidates)

    # kmeans = KMeans(n_clusters=2, random_state=0)
    clustering = SpectralClustering(n_clusters=2)
    res = clustering.fit_predict(data)
    for r, pc in zip(res, pore_candidates):
        if r:
            plt.imshow(pc)
            plt.show()
    

