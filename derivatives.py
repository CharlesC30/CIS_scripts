# functions for computing image derivatives

import numpy as np
from scipy import ndimage

def dx(im, mode="center"):
    if mode == "center":
        kx = -1 * np.array([[-1, 0, 1]])
    if mode == "forward":
        kx = -1 * np.array([[0, -1, 1]])
    if mode == "backward":
        kx = -1 * np.array([[-1, 1, 0]])
    return ndimage.convolve(im, kx)


def dy(im, mode="center"):
    if mode == "center":
        ky = -1 * np.array([[-1], [0], [1]])
    if mode == "forward":
        ky = -1 * np.array([[0], [-1], [1]])
    if mode == "backward":
        ky = -1 * np.array([[-1], [1], [0]])
    return ndimage.convolve(im, ky)


def imderiv(im, mode="center"):
    return np.sqrt(dx(im, mode=mode) ** 2 + dy(im, mode=mode) ** 2)


def imderiv_max(im, mode="center"):
    return np.maximum(np.abs(dx(im, mode=mode)), np.abs(dy(im, mode=mode)))

def full_sobel(im):
    sobel_x = ndimage.sobel(im, axis=1)
    sobel_y = ndimage.sobel(im, axis=0)
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2)