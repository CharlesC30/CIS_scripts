import numpy as np
import matplotlib.pyplot as plt
import json
import os
import cv2
from imgutils import load_and_normalize


def find_largest_connected_component_in_subregion(image, subregion):
    # Extract the specified sub-region from the image
    subregion_image = image[subregion[0]:subregion[2], subregion[1]:subregion[3]]

    # Apply CCL to the sub-region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(subregion_image, connectivity=8)

    # Find the index of the largest connected component (excluding the background)
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask for the largest component in the sub-region
    largest_component_mask = (labels == largest_component_index).astype(np.uint8)

    # Create a binary mask for the largest component in the original image
    largest_component_full_mask = np.zeros_like(image)
    largest_component_full_mask[subregion[0]:subregion[2], subregion[1]:subregion[3]] = largest_component_mask

    return largest_component_full_mask


if __name__ == "__main__":
    with open("true_pore_candidate_50-50-thresh_plus_extbbox.json", "r") as file:
        thresh5050_extbboxs = json.load(file)

    for data_name in thresh5050_extbboxs:
        image = load_and_normalize(f"pin_pore_data/{data_name}.npy", 8)
        pore_masks = []
        for thresh, bbox in thresh5050_extbboxs[data_name]:
            binary_image = image >= thresh
            # plt.imshow(binary_image)
            # plt.show()
            pore_mask = find_largest_connected_component_in_subregion(1 - binary_image.astype(np.uint8), bbox)
            # plt.imshow(p
            pore_masks.append(pore_mask)

        all_pore_mask = np.sum(pore_masks, axis=0)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image, cmap="gray")
        ax2.imshow(1 - all_pore_mask, cmap="gray")
        ax1.set_xticks([]), ax1.set_yticks([])
        ax2.set_xticks([]), ax2.set_yticks([])
        plt.savefig(f"/zhome/clarkcs/Pictures/pore_masks/{data_name}.png")

        
