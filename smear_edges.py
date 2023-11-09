import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import derivatives as drv

from imgutils import load_and_normalize

def smear_object_to_boundary(image, contour_coords):
    # idea iterate over the boundary pixels of the image and look "inwards"
    # when you find the appropriate pixel along object contour set all the pixels
    # in between to the value at the contour

    smeared_image = image.copy()
    # iterate over rows of image
    for i in range(image.shape[0]):
        coords_at_row = contour_coords[contour_coords[:, 0] == i]
        if len(coords_at_row) > 0: 
            left_contour_end = np.min(coords_at_row[:, 1])
            right_contour_end = np.max(coords_at_row[:, 1])

            left_contour_val = image[left_contour_end, i]
            right_contour_val = image[right_contour_end, i]
            smeared_image[0: left_contour_end, i] = left_contour_val
            smeared_image[right_contour_end:, i] = right_contour_val
    return smeared_image

if __name__ == "__main__":
    image = load_and_normalize("pin_pore_data/one-pin-multipore-near-edge.npy", 8)
    thresh = skimage.filters.threshold_otsu(image)
    binary_image = image >= thresh
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntrs = np.squeeze(contours)
    smeared = smear_object_to_boundary(image, cntrs)
    plt.imshow(smeared)
    plt.show()

    ss = drv.full_sobel(smeared)
    plt.imshow(ss)
    plt.show()