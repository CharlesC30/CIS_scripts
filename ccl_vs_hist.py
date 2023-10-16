# comparing ccl thresholding method vs using the image histogram
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal
from ccl_2d_thresh import ccl_threshold

def hist_threshold(img):
    hist, bin_edges = np.histogram(img, bins=256)
    peaks, _ = scipy.signal.find_peaks(hist, prominence=50)
    thresh = peaks[0] + np.argmin(hist[peaks[0]: peaks[-1]])
    return thresh


hist_thresholds = []
ccl_thresholds = []

import h5py

with h5py.File("/lhome/clarkcs/Cu-pins/high_quality/all-pins.hdf5") as f:
    for i in range(0, f["Volume"].shape[0], 1):
        print(f"looking at slice {i}")
        image = np.array(f["Volume"][i])
        cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)

        try:
            ccl_thresh = ccl_threshold(image)
        except:
            ccl_thresh = None

        try:
            hist_thresh = hist_threshold(image)
        except:
            hist_thresh = None

        ccl_thresholds.append(ccl_thresh)
        hist_thresholds.append(hist_thresh)
        print(f"ccl threshold: {ccl_thresh}, hist threshold: {hist_thresh}")

plt.plot(ccl_thresholds, ".", label="ccl")
plt.plot(hist_thresholds, ".", label="hist")
plt.legend()
plt.show()
