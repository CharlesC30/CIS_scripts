import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def interpolate_sinogram(input_sinogram, n_angles):
    x = np.arange(input_sinogram.shape[1])
    theta = np.arange(0, n_angles, n_angles // input_sinogram.shape[0])
    interp = RegularGridInterpolator((theta, x), input_sinogram, bounds_error=False, fill_value=None)
    
    xg, thetag = np.meshgrid(x, np.arange(n_angles))

    return interp((thetag, xg))


if __name__ == "__main__":
    TEST_PATH = "/lhome/clarkcs/aRTist_simulations/aRTist_train_data/sinograms/train_full_sinogram_010.npy"
    sinogram = np.squeeze(np.load(TEST_PATH))
    sparse_sinogram = sinogram[::10, :]
    interp_sinogram = interpolate_sinogram(sparse_sinogram, 1000)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(sinogram)
    ax2.imshow(sparse_sinogram)
    ax3.imshow(interp_sinogram)
    plt.show()