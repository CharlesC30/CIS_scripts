import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import h5py

IMAGE: np.ndarray
with h5py.File("/lhome/clarkcs/aRTist_simulations/254um_pixels/siemens-star_100proj_5mm-off-center_254um-pixels.hdf5") as f:
    IMAGE = np.array(f["Volume"][:, 1152 // 2, :])

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

CENTER = (1090, 800)
N = 1000
def plot_circular_profile(image, center, radius, n):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image, cmap="gray")
    pts = make_circular_sampling_points(center, radius, n)
    ax1.plot(pts[0], pts[1], "r")
    ax2.plot(interpolate_on_image(IMAGE, pts))
    fig.suptitle(f"radius = {radius * 1.72} μm")
    plt.show()

# plot_circular_profile(IMAGE, CENTER, 200, N)
for radius in range(0, 300+1, 25):
    plot_circular_profile(IMAGE, CENTER, radius, N)

