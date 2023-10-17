import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters

image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100.npy")
pin_centers = np.load("hq_Cu_pin_centers.npy")
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)

sobel_image = filters.sobel(image)
cv2.normalize(sobel_image, sobel_image, 0, 256, cv2.NORM_MINMAX)

for center in pin_centers:
    roi_x = int(center[1])
    roi_y = int(center[0])
    pin_roi = image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]
    sobel_roi = sobel_image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]

    plt.imshow(pin_roi)
    plt.show()
    plt.imshow(sobel_roi)
    plt.show()

    for thresh in range(256):
        binary_sobel = sobel_roi >= thresh
        plt.imshow(binary_sobel)
        plt.title(f"threshold={thresh}")
        plt.show()

