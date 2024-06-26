import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cc3d
import cv2
from skimage import measure
from ccl_2d_thresh import ccl_threshold

target_path = "/lhome/clarkcs/Cu-pins/low_quality"
os.chdir(target_path)

# # full ranges
# slice_ranges = {
#     2174: range(115, 215),
#     2175: range(118, 218),
#     2176: range(142, 242),
#     2189: range(79, 179),
#     2320: range(98, 198),
#     2322: range(91, 191),
#     2323: range(78, 178),
#     2324: range(114, 214),
#     2505: range(83, 183),
# }

# for testing
slice_ranges = {
    2174: range(115, 215, 49),
    2175: range(118, 218, 49),
    2176: range(142, 242, 49),
    2189: range(79, 179, 49),
    2320: range(98, 198, 49),
    2322: range(91, 191, 49),
    2323: range(78, 178, 49),
    2324: range(114, 214, 49),
    2505: range(83, 183, 49),
}

def read_dataid(filename):
    return int(filename[1:7])

# this dict is the final output you want
# filenames are the keys with pin center positions as the values
pin_centers = {}


def plot_pins_with_rois(ax, binary_pin_image):
    ax.imshow(binary_pin_image, cmap="gray")
    labels = cc3d.connected_components(binary_pin_image)
    dusted_labels = cc3d.dust(labels, threshold=100)
    regionprops = measure.regionprops(dusted_labels)
    for rp in regionprops:
        c = rp.centroid
        rect = patches.Rectangle((c[1]-128, c[0]-128), 256, 256, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    return ax


for filename in os.listdir():
    if ".hdf5" in filename:
        dataid = read_dataid(filename)
        with h5py.File(filename) as file:

            r = slice_ranges[dataid]
            mid_index = (min(r) + max(r)) // 2
            mid_image = np.array(file["Volume"][mid_index])
            cv2.normalize(mid_image, mid_image, 0, 256, cv2.NORM_MINMAX)
            
            thresh = ccl_threshold(mid_image)
            binary_image = mid_image >= thresh
            
            labels = cc3d.connected_components(binary_image)
            dusted_labels = cc3d.dust(labels, threshold=100)
            regionprops = measure.regionprops(dusted_labels)

            pin_centers[filename] = [rp.centroid for rp in regionprops]
            
            # visualize boxes around the pin centers
            for i in slice_ranges[dataid]:
                print(filename, i)
                image = np.array(file["Volume"][i])
                plt.imshow(image)
                for rp in regionprops:
                    c = rp.centroid
                    rect = patches.Rectangle((c[1]-64, c[0]-64), 128, 128, edgecolor="r", facecolor="none")
                    ax = plt.gca()
                    ax.add_patch(rect)
                plt.show()


# print(pin_centers)