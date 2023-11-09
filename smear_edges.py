import numpy as np
import matplotlib.pyplot as plt
import cv2

from imgutils import load_and_normalize

def smear_object_to_boundary(image, contour_coords):
    