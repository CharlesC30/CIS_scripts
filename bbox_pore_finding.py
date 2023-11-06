import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import cv2


from myplots import draw_bbox
from imgutils import load_and_normalize
import derivatives as drv
from PoreCandidate import PoreCandidate

def binary_regionprops(image):
    white_labels = measure.label(image)
    black_labels = measure.label(1 - image)
    white_regionprops  = measure.regionprops(white_labels)
    black_regionprops = measure.regionprops(black_labels)
    return white_regionprops, black_regionprops


def check_any_inside(target_bbox, other_bboxs, return_indices=False):
    # check if any of the "other" bboxs are contained in the target bbox
    other_bboxs = np.array(other_bboxs)
    if other_bboxs.size == 0:
        if return_indices:
            return False, np.array([])
        else:
            return False
    target_bbox = np.array(target_bbox)
    check_mins = other_bboxs[:, 0:2] >= target_bbox[0:2]
    check_maxs = other_bboxs[:, 2:4] <= target_bbox[2:4]
    check_combined = np.column_stack((check_mins, check_maxs))
    in_boundaries = np.all(check_combined, axis=1)
    if return_indices:
        return np.any(in_boundaries), np.where(in_boundaries)[0]
    else:
        return np.any(in_boundaries)


def try_all_thresholds(image):
    # loop over all possible thresholds for image
    max_val = int(np.ceil(image.max()))

    pore_candidates = []
    
    for thresh in range(0, max_val, 5):
        print(thresh)
        binary_image = image >= thresh
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image, cmap="gray")   
        ax2.imshow(binary_image, cmap="gray")
        white_rps, black_rps = binary_regionprops(binary_image)
        white_bboxs = [wrp.bbox for wrp in white_rps]
        black_bboxs = [brp.bbox for brp in black_rps]
        white_bbox_areas = [wrp.area_bbox for wrp in white_rps]
        largest_white_area = max(white_bbox_areas)

        new_candidate_bboxs = []
        for w_bbox, w_bbox_area in zip(white_bboxs, white_bbox_areas):
            if w_bbox_area == largest_white_area:
                draw_bbox(w_bbox, ax2, "yellow")
            else:
                if check_any_inside(w_bbox, black_bboxs):
                    new_candidate_bboxs.append(w_bbox) 
                    draw_bbox(w_bbox, ax2, "orange")
                else:
                    draw_bbox(w_bbox, ax2, "red")
        for existing_candidate in pore_candidates:
            if existing_candidate.exists:
                still_exists, new_bbox_idx = check_any_inside(existing_candidate.get_current_bbox(), new_candidate_bboxs, return_indices=True)
                assert len(new_bbox_idx) <= 1, "multiple new candidates inside existing pore"
                if still_exists:
                    new_bbox = new_candidate_bboxs.pop(new_bbox_idx[0])
                    existing_candidate.update_bbox(thresh, new_bbox)
                else:
                    existing_candidate.end()
        for bbox in new_candidate_bboxs:
            pc = PoreCandidate(t0=thresh, bbox0=bbox)
            pore_candidates.append(pc)

        for b_bbox in black_bboxs:
            draw_bbox(b_bbox, ax2, "green")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    image = load_and_normalize("ict_pin_pore.npy", 8)
    sobel_image = drv.full_sobel(image)
    try_all_thresholds(sobel_image)

