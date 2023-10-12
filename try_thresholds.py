import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters

image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100.npy")
pin_centers = np.load("hq_Cu_pin_centers.npy")
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)
# sobel_image = filters.sobel(image)
# plt.imshow(sobel_image)
# plt.show()

for center in pin_centers:
    roi_x = int(center[1])
    roi_y = int(center[0])
    pin_roi = image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]

    sobel_roi = filters.sobel(pin_roi)
    plt.imshow(sobel_roi)
    plt.show()

    for thresh in np.arange(0, 256, 0.5):
        binary_sobel = sobel_roi >= thresh
        print(thresh)
        plt.imshow(binary_sobel)
        plt.show()

    # high = filters.threshold_otsu(sobel_roi)
    # low = filters.threshold_triangle(sobel_roi)

    # hyst = filters.apply_hysteresis_threshold(sobel_roi, low, high)

    # plt.subplot(1, 4, 1)
    # plt.imshow(sobel_roi)
    # plt.subplot(1, 4, 2)
    # plt.imshow(sobel_roi >= low)
    # plt.subplot(1, 4, 3)
    # plt.imshow(sobel_roi >= high)
    # plt.subplot(1, 4, 4)
    # plt.imshow(hyst)
    # plt.show()


# for thresh in range(256):
#     binary_sobel = sobel_image >= thresh
#     plt.imshow(binary_sobel)
#     plt.show()
