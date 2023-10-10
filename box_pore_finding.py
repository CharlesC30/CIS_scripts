import numpy as np
import matplotlib.pyplot as plt
from twopass_ccl import twopass_ccl
from skimage import filters

image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100.npy")
pin_centers = np.load("hq_Cu_pin_centers.npy")

print(pin_centers)

for center in pin_centers:
    roi_x = int(center[1])
    roi_y = int(center[0])
    pin_roi = image[roi_y-100: roi_y+100, roi_x-100: roi_x+100]

    pin_roi_thresh = filters.threshold_otsu(pin_roi)
    binary_pin_roi = pin_roi <= pin_roi_thresh

    sobel_roi = filters.sobel(pin_roi)
    sobel_thresh = filters.threshold_otsu(sobel_roi)
    binary_sobel_roi = sobel_roi <= sobel_thresh

    # labeled_roi = twopass_ccl(pin_roi, foreground=1)

    plt.subplot(2, 2, 1)
    plt.imshow(pin_roi)
    plt.subplot(2, 2, 2)
    plt.imshow(binary_pin_roi)
    plt.subplot(2, 2, 3)
    plt.imshow(sobel_roi)
    plt.subplot(2, 2, 4)
    plt.imshow(binary_sobel_roi)
    plt.show()
