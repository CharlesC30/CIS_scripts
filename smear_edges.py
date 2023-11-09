import numpy as np
import matplotlib.pyplot as plt
import cv2

from imgutils import load_and_normalize

def smear_object_to_boundary(image, contour_coords):
    # idea iterate over the boundary pixels of the image and look "inwards"
    # when you find the appropriate pixel along object contour set all the pixels
    # in between to the value at the contour
    pass