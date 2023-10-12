import numpy as np
import matplotlib.pyplot as plt
import skimage

# image = np.load("multi_ccl_test.npy")

def twopass_ccl(image, foreground=1):
    labels, n_labels = skimage.measure.label(image, return_num=True)
    invlabels = skimage.measure.label(image, background=foreground)
    invlabels[np.nonzero(invlabels)] += n_labels
    return labels + invlabels
