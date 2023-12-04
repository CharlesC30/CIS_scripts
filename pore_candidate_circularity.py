import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import skimage

from imgutils import load_and_normalize
from myplots import draw_bbox

def calculate_circularity(contour):
    # Calculate circularity using the formula (4 * pi * area) / perimeter**2
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter**2)
    return circularity

if __name__ == "__main__":
    for image_path in os.listdir("pin_pore_data"):
        print(image_path)
        data_name = image_path[:-4]
        image = load_and_normalize(f"pin_pore_data/{image_path}", 8)
        with open(f"pcs_maxthresh_bbox/{data_name}_pcs.json", "r") as file:
            pore_candidates = json.load(file)

        avg_circularities = []
        for i, (_, bbox) in enumerate(pore_candidates):
            bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            # thresh = bbox_roi.min() + (bbox_roi.max() - bbox_roi.min()) / 2
            thresh = skimage.filters.threshold_otsu(bbox_roi)
            binary_bbox_roi = bbox_roi >= thresh

            contours, _ = cv2.findContours((1 - binary_bbox_roi.astype(np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            circularities = [calculate_circularity(cntr) for cntr in contours]
            avg_circularity = np.mean(circularities)
            avg_circularities.append(avg_circularity)
            
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image, cmap="gray")
            draw_bbox(bbox, ax1, "orange")
            ax2.imshow(binary_bbox_roi, cmap="gray")
            fig.suptitle(f"average circularity = {avg_circularity}")
            if avg_circularity >= 0.80:
                plt.savefig(f"/zhome/clarkcs/Pictures/pore_circularities/true_pores/{data_name}_{i}")
            else:
                plt.savefig(f"/zhome/clarkcs/Pictures/pore_circularities/false_pores/{data_name}_{i}")
            plt.close(fig)
        plt.plot(avg_circularities, "o")
        plt.show()