import numpy as np
import matplotlib.pyplot as plt
import cc3d
from skimage import measure
from scipy import spatial

image = np.load("/lhome/clarkcs/Cu-pins/low_quality/slice30_thresh.npy")
labels = cc3d.connected_components(image)
labels_dusted = cc3d.dust(labels, threshold=100)

# kinda unnecessary, just use skimage.measure.regionprops
def find_component_centers(labels, plot_results=False):
    centers = []
    for lbl in np.unique(labels):
        if lbl == 0:
            pass
        else:
            single_label = labels.copy()
            single_label[labels != lbl] = 0
            c = measure.centroid(single_label)
            centers.append(c)
    if plot_results:
        plt.imshow(labels)
        plt.plot([c[1] for c in centers], [c[0] for c in centers], "r*")
        plt.show()
    return centers

centers = find_component_centers(labels_dusted, plot_results=True)

def merge_nearby_points(points, distance):
    merged_points = points.copy()
    point_tree = spatial.KDTree(points)
    nearby_pairs = point_tree.query_pairs(distance, output_type="ndarray")
    for pair in nearby_pairs:
        pt0 = points[pair[0]]
        pt1 = points[pair[1]]
        merged_points.append(np.mean([pt0, pt1], axis=0))
    for i in np.sort(np.ravel(nearby_pairs))[::-1]:
        merged_points.pop(i)
    return merged_points

# merged_centers = merge_nearby_points(centers, 100)
# plt.imshow(image)
# plt.plot([c[1] for c in merged_centers], [c[0] for c in merged_centers], "r*")
# plt.show()
