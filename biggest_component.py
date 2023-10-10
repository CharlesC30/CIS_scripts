import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import skimage

vol_path = "/lhome/clarkcs/bottle-ipa/R000476-raw-down4-sparse256-reco/volume.hdf5"

slice_n = 289

with h5py.File(vol_path, "r") as file:
    vol_array = np.array(file["Volume"])

original_image = vol_array[:, slice_n, :]
cv2.normalize(original_image, original_image, 0, 65535, cv2.NORM_MINMAX)
threshold_image = original_image.copy()

# try all possible thresholds and plot results
largest_component_sizes = []
number_of_components = []
threshold_range = range(0, 65535+1, 85)
for threshold in threshold_range:
    print(threshold)
    threshold_image[original_image < threshold] = 0
    threshold_image[original_image >= threshold] = 1
    connected_components, N = skimage.measure.label(threshold_image, return_num=True)

    largest_size = np.max(np.bincount(connected_components.flat)[1:])
    largest_component_sizes.append(largest_size)
    number_of_components.append(N)

# plt.plot(threshold_range, largest_component_sizes)
plt.plot(threshold_range, number_of_components)
plt.show()

# # display thresholded image and connected components at certain threshold value
threshold = 0
threshold_image[original_image < threshold] = 0
threshold_image[original_image >= threshold] = 1
connected_components = skimage.measure.label(threshold_image)
plt.imshow(threshold_image, cmap="binary")
plt.show()
plt.imshow(skimage.color.label2rgb(connected_components))
plt.show()
