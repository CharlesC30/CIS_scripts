import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import cv2
import os

os.chdir("/lhome/clarkcs/Cu-pins/pin-pore-examples")
image = np.load("ict_pin_pore.npy")
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)

# def ccl_thresh(image):
