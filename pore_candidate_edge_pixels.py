import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2

from imgutils import load_and_normalize
from myplots import draw_bbox

def find_largest_connected_component(image):
    # Apply CCL
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)

    # Find the index of the largest connected component (excluding the background)
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask for the largest component
    largest_component_mask = (labels == largest_component_index).astype(np.uint8)

    return largest_component_mask


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

def extend_bbox(bbox, binary_bbox_roi, image_shape):
    top_row = binary_bbox_roi[0, :]
    new_min_row = bbox[0] - np.count_nonzero(1 - top_row)
    if new_min_row < 0:
        new_min_row = 0
    left_col = binary_bbox_roi[:, 0]
    new_min_col = bbox[1] - np.count_nonzero(1 - left_col)
    if new_min_col < 0:
        new_min_col = 0
    bottom_row = binary_bbox_roi[-1, :]
    new_max_row = bbox[2] + np.count_nonzero(1 - bottom_row)
    if new_max_row >= image_shape[0]:
        new_max_row = image_shape[0] - 1
    right_col = binary_bbox_roi[:, -1]
    new_max_col = bbox[3] + np.count_nonzero(1 - right_col)
    if new_max_col >= image_shape[1]:
        new_max_col = image_shape[1] - 1
    return (new_min_row, new_min_col, new_max_row, new_max_col)


if __name__ == "__main__":
    for image_path in os.listdir("pin_pore_data"):
        print(image_path)
        data_name = image_path[:-4]
        image = load_and_normalize(f"pin_pore_data/{image_path}", 8)
        with open(f"pcs_maxthresh_bbox/{data_name}_pcs.json", "r") as file:
            pore_candidates = json.load(file)

        os.chdir("/zhome/clarkcs/Pictures/extending_bbox_check_edges_50-50-thresh")
        # if not os.path.exists(f"{data_name}"):
        #     os.mkdir(f"{data_name}")
        for i, (_, bbox) in enumerate(pore_candidates):
            bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            bbox_thresh = find_half_thresh(bbox_roi)
            # bbox_thresh = (bbox_roi.max() - bbox_roi.min()) / 4 + bbox_roi.min()
            # bbox_thresh = bbox_roi.min() + 1
            binary_roi = bbox_roi >= bbox_thresh
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image, cmap="gray")
            draw_bbox(bbox, ax1, "orange")

            ax2.imshow(binary_roi, cmap="gray")
            if any_black_boundary(binary_roi):
                new_bbox = extend_bbox(bbox, binary_roi, image.shape)
                new_bbox_roi = image[new_bbox[0]: new_bbox[2], new_bbox[1]: new_bbox[3]]
                new_binary_roi = new_bbox_roi >= bbox_thresh
                print(new_bbox)
                ax2.clear()
                ax2.imshow(new_binary_roi, cmap="gray")
                largest_component = find_largest_connected_component(1 - new_binary_roi.astype(np.uint8))
                if any_black_boundary(1 - largest_component):
                    plt.savefig(f"false_pores/{data_name}_pore_candidate_{i}.png")
                else:
                    plt.savefig(f"true_pores/{data_name}_pore_candidate_{i}.png")
            else:
                plt.savefig(f"true_pores/{data_name}_pore_candidate_{i}.png")
            plt.close(fig)
        os.chdir("/zhome/clarkcs/scripts")