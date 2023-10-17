import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, filters
from ccl_2d_thresh import ccl_threshold

def find_pores(image):
    plt.imshow(image)
    plt.show()
    sobel_image = filters.sobel(image)
    binary_sobel = (sobel_image > 0).astype(int)
    plt.imshow(binary_sobel)
    plt.show()
    labels = measure.label(binary_sobel, background=-1)

    plt.imshow(labels)
    plt.show()

    regionprops = measure.regionprops(labels)
    for rp in regionprops:
        roi = rp.slice
        plt.imshow(labels[roi])
        plt.show()


image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100.npy")
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)
# thresh = ccl_threshold(image)
# print(thresh)
thresh = 142
image = image >= thresh
pin_centers = np.load("hq_Cu_pin_centers.npy")
c = pin_centers[0]
roi_x = int(c[1])
roi_y = int(c[0])
pin_roi = image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]

find_pores(pin_roi)
