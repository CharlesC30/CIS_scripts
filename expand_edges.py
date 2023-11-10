import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology
import cv2
from ccl_2d_thresh import ccl_threshold

from imgutils import load_and_normalize
import derivatives as drv

def expand_edges(image, edge_coords):
    expanded = np.zeros(shape=image.shape)
    for x, y in edge_coords:
        expanded[y, x] = image[y, x]
    expanded = morphology.dilation(expanded, morphology.square(3))
    # plt.imshow(expanded)
    # plt.show()
    return np.maximum(image, expanded)

if __name__ == "__main__":
    plt.gray()
    image = load_and_normalize("pin_pore_data/one-pin-multipore-near-edge.npy", 8)
    thresh = ccl_threshold(image)
    binary_image = image >= thresh
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntrs = np.squeeze(contours)
    expanded = expand_edges(image, cntrs)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(image)
    ax2.imshow(expanded)
    ax3.imshow(drv.full_sobel(expanded))
    plt.show()

