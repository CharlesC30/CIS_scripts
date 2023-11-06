import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import cv2


from myplots import draw_bbox
from imgutils import load_and_normalize
import derivatives as drv

def binary_regionprops(image):
    white_labels = measure.label(image)
    black_labels = measure.label(1 - image)
    white_regionprops  = measure.regionprops(white_labels)
    black_regionprops = measure.regionprops(black_labels)
    return white_regionprops, black_regionprops


def check_pore_candidates(white_bbox, black_bboxs):
    # check if any of the black bboxs are contained in the white bbox
    np.array(black_bboxs)
    return np.any([])


def try_all_thresholds(image):
    # loop over all possible thresholds for image
    max_val = int(np.ceil(image.max()))
    for thresh in range(max_val):
        print(thresh)
        binary_image = image >= thresh
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image, cmap="gray")   
        ax2.imshow(binary_image, cmap="gray")
        white_rps, black_rps = binary_regionprops(binary_image)
        for wrp in white_rps:
            draw_bbox(wrp.bbox, ax2, "red")
        for brp in black_rps:
            draw_bbox(brp.bbox, ax2, "green")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    image = load_and_normalize("ict_pin_pore.npy", 8)
    sobel_image = drv.full_sobel(image)
    try_all_thresholds(sobel_image)

