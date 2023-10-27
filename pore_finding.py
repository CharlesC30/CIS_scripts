import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage import measure, filters
from ccl_2d_thresh import ccl_threshold

def draw_bbox(bbox):
    min_y, min_x, max_y, max_x = bbox
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle((min_x, min_y), width, height, edgecolor="r", facecolor="none")
    ax = plt.gca()
    ax.add_patch(rect)


def find_pores(image):
    # plt.imshow(image)
    # plt.show()
    sobel_image = filters.sobel(image)
    binary_sobel = (sobel_image > 0).astype(int)
    # plt.imshow(binary_sobel)
    # plt.show()
    labels = measure.label(binary_sobel, background=-1)

    # plt.imshow(labels)
    # plt.show()

    regionprops = measure.regionprops(labels)
    for rp in regionprops:
        if rp.area_bbox < 5_000:
            roi = labels[rp.slice]
            if len(np.unique(roi)) >= 3:
                print("pore found?")
                plt.subplot(1,2,1)
                plt.imshow(sobel_image)
                draw_bbox(rp.bbox)
                plt.subplot(1,2,2)
                plt.imshow(labels)
                draw_bbox(rp.bbox)
                plt.show()


image = np.load("/lhome/clarkcs/Cu-pins/high_quality/slice100.npy")
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)
thresh = ccl_threshold(image)
print(thresh)
thresh = 142
image = image >= thresh
pin_centers = np.load("hq_Cu_pin_centers.npy")

for c in pin_centers:
    roi_x = int(c[1])
    roi_y = int(c[0])
    pin_roi = image[roi_y-128: roi_y+128, roi_x-128: roi_x+128]

    find_pores(pin_roi)
