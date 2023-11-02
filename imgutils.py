import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_and_normalize(image_path, n_bits):
    image = np.load(image_path)
    cv2.normalize(image, image, 0, 2 ** n_bits, cv2.NORM_MINMAX)
    return image