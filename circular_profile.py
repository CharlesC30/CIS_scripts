import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import h5py

IMAGE_1: np.ndarray
IMAGE_2: np.ndarray
# with h5py.File("/lhome/clarkcs/aRTist_simulations/254um_pixels/siemens-star_100proj_5mm-off-center_254um-pixels.hdf5") as f:
#     IMAGE = np.array(f["Volume"][:, 1152 // 2, :])
IMAGE_1 = np.load("/lhome/clarkcs/aRTist_simulations/cylinder/lama_inpainting/ground_truth.npy")
IMAGE_2 = np.load("/lhome/clarkcs/aRTist_simulations/cylinder/lama_inpainting/nn_prediction.npy")

def make_circular_sampling_points(c, r, n):
    """
    `c`: center of circle (x, y)
    `r`: radius
    `n`: number of sampling points
    """
    angles = np.linspace(0, 2 * np.pi, n)
    return (r * np.cos(angles) + c[0], r * np.sin(angles) + c[1])


def interpolate_on_image(image, points):
    xs = np.linspace(0, image.shape[1] - 1, image.shape[1])
    ys = np.linspace(0, image.shape[0] - 1, image.shape[0])
    interp = RegularGridInterpolator((xs, ys), image)
    return interp(np.array([points[1], points[0]]).T)

def plot_circular_profile(image, center, radius, n, ax_image, ax_plot):
    ax_image.imshow(image, cmap="gray")
    pts = make_circular_sampling_points(center, radius, n)
    ax_image.plot(pts[0], pts[1], "r")
    ax_plot.plot(interpolate_on_image(image, pts), "*")
    ax_plot.plot(interpolate_on_image(image, pts), "-")

CENTER_1 = (IMAGE_1.shape[0] // 2, IMAGE_1.shape[1] // 2)
CENTER_2 = (IMAGE_2.shape[0] // 2, IMAGE_2.shape[1] // 2)
N = 1000
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plot_circular_profile(IMAGE_1, CENTER_1, 180, N, ax1, ax3)
plot_circular_profile(IMAGE_2, CENTER_2, 180, N, ax2, ax3)
plt.show()
# for radius in range(0, 300+1, 25):
#     plot_circular_profile(IMAGE, CENTER, radius, N)

