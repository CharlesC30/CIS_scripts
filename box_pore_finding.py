import numpy as np
import matplotlib.pyplot as plt
from twopass_ccl import twopass_ccl
from ccl_2d_thresh import ccl_threshold
import cc3d
import cv2
from skimage import filters

image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100.npy")
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)
binary_image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100_thresh.npy")
pin_centers = np.load("hq_Cu_pin_centers.npy")

# sobel_image = filters.sobel(image)
# sobel_thresh = ccl_threshold(sobel_image)

for center in pin_centers:
    roi_x = int(center[1])
    roi_y = int(center[0])
    pin_roi = image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]
    binary_pin_roi = binary_image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]

    sobel_roi = filters.sobel(pin_roi)
    binary_sobel_roi = filters.sobel(binary_pin_roi)
    binary_sobel_roi = binary_sobel_roi > 0

    # labeled_binary_sobel = twopass_ccl(binary_sobel_roi)

    # plt.imshow(labeled_binary_sobel)
    # plt.show()

    plt.subplot(2, 2, 1)
    plt.imshow(pin_roi)
    plt.subplot(2, 2, 2)
    plt.imshow(binary_pin_roi)
    plt.subplot(2, 2, 3)
    plt.imshow(sobel_roi)
    plt.subplot(2, 2, 4)
    plt.imshow(binary_sobel_roi)
    plt.show()
