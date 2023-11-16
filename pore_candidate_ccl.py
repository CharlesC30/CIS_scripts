import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
from skimage import measure

from myplots import draw_bbox
from imgutils import load_and_normalize

if __name__ == "__main__":
    for filename in os.listdir("pin_pore_data"):
        os.chdir("/zhome/clarkcs/scripts")
        data_name = Path(filename).stem
        image = load_and_normalize(f"pin_pore_data/{filename}", 8)
        with open(f"pcs_maxthresh_bbox/{data_name}_pcs.json") as pc_file:
            pore_candidates = json.load(pc_file)

        if not os.path.exists(f"/zhome/clarkcs/Pictures/pore_candidate_ccl/{data_name}"):
            os.makedirs(f"/zhome/clarkcs/Pictures/pore_candidate_ccl/{data_name}")
        os.chdir(f"/zhome/clarkcs/Pictures/pore_candidate_ccl/{data_name}")

        for i, (_, bbox) in enumerate(pore_candidates):
            if not os.path.exists(f"pore_candidate_{i}"):
                os.mkdir(f"pore_candidate_{i}")

            bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            
            label_count = []
            threshed_bboxs = []
            for thresh in range(int(bbox_roi.min()), int(bbox_roi.max() + 1)):
                binary_bbox_roi = bbox_roi >= thresh
                threshed_bboxs.append((thresh, binary_bbox_roi))
                bbox_labels, n_labels = measure.label(1 - binary_bbox_roi, return_num=True)
                label_count.append(n_labels)
            print(label_count)

            for thresh, binary_bbox in threshed_bboxs:
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(image, cmap="gray")
                draw_bbox(bbox, ax1, "orange")
                ax2.imshow(binary_bbox, cmap="gray")
                ax3.plot([t for t, _ in threshed_bboxs], label_count)
                # arrow_height = max(label_count) / 25
                # arrow = patches.Arrow(thresh, min(label_count)-arrow_height, 0, arrow_height, color="red")
                # ax3.add_patch(arrow)
                ax3.axvline(thresh, c="red")

                plt.savefig(f"pore_candidate_{i}/thresh_{thresh:0{4}}.png")
                # plt.show()
                plt.close(fig)

