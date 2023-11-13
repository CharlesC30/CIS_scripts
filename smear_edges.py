import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import derivatives as drv

from imgutils import load_and_normalize

# def set_edge_to_max(image, edge_coords, window_size=5):
#     modified_image = image.copy()
#     if window_size % 2 != 1:
#         raise ValueError("window size must be odd")
#     window_step = window_size // 2
#     for x, y in edge_coords:
#         image_window = image[y-window_step: y+window_step+1, x-window_step: x+window_step+1]
#         window_max = np.max(image_window)
#         modified_image[y, x] = window_max
#     return modified_image

def set_edge_to_max(image, edge_coords, window_size=5):
    modified_image = image.copy()
    if window_size % 2 != 1:
        raise ValueError("window size must be odd")
    window_step = window_size // 2
    for x, y in edge_coords:
        image_window = image[y-window_step: y+window_step+1, x-window_step: x+window_step+1]
        window_max = np.max(image_window)
        modified_image[y-window_step: y+window_step+1, x-window_step: x+window_step+1] = window_max
    return modified_image


def find_edge_coords(image):
    thresh = skimage.filters.threshold_otsu(image)
    binary_image = image >= thresh
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.squeeze(contours)


def smear_object_to_boundary(image, contour_coords, axis="both"):
    # idea iterate over the boundary pixels of the image and look "inwards"
    # when you find the appropriate pixel along object contour set all the pixels
    # in between to the value at the contour

    smeared_image = image.copy()
    if axis == "both" or axis == "y":
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
    if axis == "both" or axis == "x":
        # iterate over columns
        for j in range(image.shape[1]):
            coords_at_col = contour_coords[contour_coords[:, 1] == j]
            if len(coords_at_col) > 0: 
                top_contour_end = np.min(coords_at_col[:, 0])
                bottom_contour_end = np.max(coords_at_col[:, 0])

                top_contour_val = image[j, top_contour_end]
                bottom_contour_val = image[j, bottom_contour_end]
                smeared_image[j, 0: top_contour_end] = top_contour_val
                smeared_image[j, bottom_contour_end:] = bottom_contour_val
        
    return smeared_image

if __name__ == "__main__":
    plt.gray()
    image = load_and_normalize("pin_pore_data/one-pin-multipore-near-edge.npy", 8)
    plt.imshow(image)
    plt.show()

    cntrs = find_edge_coords(image)
    image = set_edge_to_max(image, cntrs)
    plt.imshow(image)
    plt.show()
    smeared1 = smear_object_to_boundary(image, cntrs, axis='x')

    thresh1 = skimage.filters.threshold_otsu(smeared1)
    binary_smeared1 = smeared1 >= thresh1
    contours, _ = cv2.findContours(binary_smeared1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntrs = np.squeeze(contours)
    smeared = smear_object_to_boundary(smeared1, cntrs, axis='y')

    plt.imshow(smeared)
    plt.show()

    drv_image = drv.full_sobel(smeared)
    plt.imshow(drv_image)
    plt.show()