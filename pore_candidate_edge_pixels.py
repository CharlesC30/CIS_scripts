import numpy as np
import matplotlib.pyplot as plt
import os
import json

from imgutils import load_and_normalize
from myplots import draw_bbox

def find_half_thresh(image):
    for thresh in range(int(image.min()), int(image.max()+1)):
        binary_image = image >= thresh
        num_black_pixels = np.count_nonzero(1 - binary_image)
        if num_black_pixels >= image.size / 2:
            break
    return thresh

def any_black_boundary(image):
    boundary_vals = np.concatenate([image[:, 0], image[:, -1], image[0, :], image[-1, :]])
    return np.any(1 - boundary_vals)


if __name__ == "__main__":
    for image_path in os.listdir("pin_pore_data"):
        print(image_path)
        data_name = image_path[:-4]
        image = load_and_normalize(f"pin_pore_data/{image_path}", 8)
        with open(f"pcs_maxthresh_bbox/{data_name}_pcs.json", "r") as file:
            pore_candidates = json.load(file)

        os.chdir("/zhome/clarkcs/Pictures/pore_candidate_edge_check_lowest-thresh")
        # if not os.path.exists(f"{data_name}"):
        #     os.mkdir(f"{data_name}")
        for i, (_, bbox) in enumerate(pore_candidates):
            bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            # bbox_thresh = find_half_thresh(bbox_roi)
            bbox_thresh = int(bbox_roi.min() + 1)
            binary_roi = bbox_roi >= bbox_thresh
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image, cmap="gray")
            draw_bbox(bbox, ax1, "orange")

            ax2.imshow(binary_roi, cmap="gray")
            if any_black_boundary(binary_roi):
                plt.savefig(f"false_pores/{data_name}_pore_candidate_{i}.png")
            else:
                plt.savefig(f"true_pores/{data_name}_pore_candidate_{i}.png")
            plt.close(fig)
        os.chdir("/zhome/clarkcs/scripts")