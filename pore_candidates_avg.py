import numpy as np
import matplotlib.pyplot as plt
import os
import json

from imgutils import load_and_normalize
from myplots import draw_bbox

if __name__ == "__main__":
    for image_path in os.listdir("pin_pore_data"):
        print(image_path)
        data_name = image_path[:-4]
        image = load_and_normalize(f"pin_pore_data/{image_path}", 8)
        with open(f"pcs_maxthresh_bbox/{data_name}_pcs.json", "r") as file:
            pore_candidates = json.load(file)

        os.chdir("/zhome/clarkcs/Pictures/pore_candidate_avg")
        if not os.path.exists(f"{data_name}"):
            os.mkdir(f"{data_name}")
        for i, (_, bbox) in enumerate(pore_candidates):
            bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]

            avg_roi_value = np.mean(bbox_roi)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image)
            draw_bbox(bbox, ax1, "orange")

            ax2.hist(image.ravel(), bins=256)
            ax2.axvline(avg_roi_value, c="r")
            plt.savefig(f"{data_name}/pore_candidate_{i}.png")
            plt.close(fig)
        os.chdir("/zhome/clarkcs/scripts")
            