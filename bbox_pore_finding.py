import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os
import json

from myplots import draw_bbox
from imgutils import load_and_normalize
import derivatives as drv
from PoreCandidate import PoreCandidate
from smear_edges import smear_object_to_boundary, set_edge_to_max, find_edge_coords

def binary_regionprops(image, connectivity=None):
    white_labels = measure.label(image, connectivity=connectivity)
    black_labels = measure.label(1 - image, connectivity=connectivity)
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


def remove_material_boundary(image):
    modified_image = image.copy()
    cntrs0 = find_edge_coords(modified_image)
    modified_image = set_edge_to_max(modified_image, cntrs0)
    modified_image = smear_object_to_boundary(modified_image, cntrs0, "x")
    cntrs1 = find_edge_coords(modified_image)
    modified_image = smear_object_to_boundary(modified_image, cntrs1, "y")
    return modified_image


def calc_bbox_area(bbox):
    return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def try_all_thresholds(image: np.ndarray, save_plots=False, ignore_largest_bbox=True) -> list[PoreCandidate]:
    # loop over all possible thresholds for image
    max_val = int(np.ceil(image.max()))

    pore_candidates = []
    
    for thresh in range(0, max_val, 1):
        print(thresh)
        binary_image = image >= thresh
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(binary_image, cmap="gray")   
        ax2.imshow(binary_image, cmap="gray")
        white_rps, black_rps = binary_regionprops(binary_image)
        white_bboxs = [wrp.bbox for wrp in white_rps]
        black_bboxs = [brp.bbox for brp in black_rps]
        white_bbox_areas = [wrp.area_bbox for wrp in white_rps]
        largest_white_area = max(white_bbox_areas)

        new_candidate_bboxs = []
        for w_bbox, w_bbox_area in zip(white_bboxs, white_bbox_areas):
            if w_bbox_area == largest_white_area and ignore_largest_bbox:
                draw_bbox(w_bbox, ax2, "yellow", **{"linewidth": 1})
            else:
                any_inside, inside_bbox_idxs = check_any_inside(w_bbox, black_bboxs, return_indices=True)
                if any_inside:
                    inside_bboxs = [black_bboxs[idx] for idx in inside_bbox_idxs]
                    new_candidate_bboxs.append({"outer": w_bbox, "inner": inside_bboxs}) 
                    draw_bbox(w_bbox, ax2, "orange", **{"linewidth": 1})
                else:
                    draw_bbox(w_bbox, ax2, "red", **{"linewidth": 1})
        for candidate in pore_candidates:
            if candidate.exists:
                still_exists, contained_bbox_idxs = check_any_inside(candidate.get_current_outer_bbox(), 
                                                                     [ncb["outer"] for ncb in new_candidate_bboxs], 
                                                                     return_indices=True)
                if still_exists:
                    new_bbox_idx = max(contained_bbox_idxs, key=lambda idx: calc_bbox_area(new_candidate_bboxs[idx]["outer"]))
                    new_bbox = new_candidate_bboxs.pop(new_bbox_idx)
                    candidate.update_bbox(thresh, new_bbox)
                else:
                    candidate.end()
        for bbox in new_candidate_bboxs:
            pc = PoreCandidate(t0=thresh, bbox0=bbox)
            pore_candidates.append(pc)

        for b_bbox in black_bboxs:
            draw_bbox(b_bbox, ax2, "green")

        if save_plots:
            fig.suptitle(f"threshold={thresh}")
            if not os.path.exists("thresh_images"):
                os.mkdir("thresh_images")
            fig.savefig(f"thresh_images/thresh_{thresh:0{9}}", dpi=200)
        plt.close(fig)

    # plot the lifetimes of each "pore"
    if save_plots:
        fig, ax = plt.subplots()
        max_thresh = max((pc.max_threshold for pc in pore_candidates))
        ax.set_xlim(0, max_thresh + 10)
        ax.set_xticks(np.arange(0, max_thresh+10, 10))
        for i, pc in enumerate(pore_candidates):
            ax.barh(i, width=(pc.max_threshold - pc.min_threshold), left=pc.min_threshold)
        ax.grid(True, axis='x')
        ax.set_xlabel("threshold")
        fig.savefig("pore_lifetimes")
        plt.close(fig)
    return pore_candidates


def check_bbox_histogram(image, bbox):
    bbox_roi = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(image)
    draw_bbox(bbox, ax1, "orange")
    ax2.imshow(bbox_roi)
    ax3.hist(np.ravel(bbox_roi))
    plt.show()


if __name__ == "__main__":

    for i, image_path in enumerate(os.listdir("pin_pore_data")):
        print(image_path)
        data_name = image_path[:-4]
        image = load_and_normalize(f"pin_pore_data/{image_path}", 16)
        derivative_image = drv.full_sobel(image)
        os.chdir("/lhome/clarkcs/pore_detection_16bit")
        if not os.path.exists(f"{data_name}"):
            os.mkdir(f"{data_name}")
        os.chdir(f"{data_name}")
        pore_candidates = try_all_thresholds(derivative_image, save_plots=True, ignore_largest_bbox=True)
        os.chdir("/zhome/clarkcs/scripts/pcs_inner_outer_bboxs_16bit")
        with open(f"{data_name}_pcs_16bit.json", "w") as file:
            json.dump([(pc.max_threshold, pc.get_current_bboxs()) for pc in pore_candidates], file)
        os.chdir("/zhome/clarkcs/scripts")

