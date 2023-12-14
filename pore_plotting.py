import numpy as np
import matplotlib.pyplot as plt
import json

from myplots import draw_bbox
from imgutils import load_and_normalize

def plot_bbox_roi_threshed(image, bbox, threshold):
    fig, axs = plt.subplots(1, 3)
    for ax in axs:
        ax.set_xticks([]), ax.set_yticks([])
    axs[0].imshow(image, cmap="gray")
    draw_bbox(bbox, axs[0], color="orange", **{"linewidth": 2})
    bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    axs[1].imshow(bbox_roi, cmap="gray")
    axs[2].imshow((bbox_roi >= threshold).astype(np.uint8), cmap="gray")

if __name__ == "__main__":
    with open("true_pore_candidate_50-50-thresh_plus_extbbox.json", "r") as file:
        pores_thresh_bboxs = json.load(file)
    
    for data_name in pores_thresh_bboxs:
        im = load_and_normalize(f"pin_pore_data/{data_name}.npy", 8)
        for i, (thresh, bbox) in enumerate(pores_thresh_bboxs[data_name]):
            plot_bbox_roi_threshed(im, bbox, thresh)
            plt.savefig(f"/zhome/clarkcs/Pictures/pore_roi_thresh/{data_name}_{i}.png")