"""
script to test the effect of applying ramp filter (in freq. domain) to a single projection
"""
import numpy as np
import matplotlib.pyplot as plt
import os
# from skimage.transform.radon_transform import _get_fourier_filter

os.chdir("/lhome/clarkcs")
filename = "first_cylinder_proj.npy"
projection = np.load(filename)

def ramp_filter(size):
    return np.concatenate(
        (
            np.arange(0, size/2, 1, dtype=int),
            np.arange(size/2, 0, -1, dtype=int)
        )
    )

def ramp_filter_fancy(size):
    """stolen from: 
    https://ciip.in.tum.de/elsadocs/guides/python_guide/filtered_backprojection.html"""
    n = np.concatenate(
        (
            # increasing range from 1 to size/2, and again down to 1, step size 2
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    return np.transpose(2 * np.real(np.fft.fft(f))[:, np.newaxis])

def apply_cutoff(rampfilter, cutoff):
    rampfilter_cut = rampfilter.copy()
    rampfilter_cut[rampfilter > cutoff] = 0
    return rampfilter_cut

padding = np.zeros(projection.shape)
padded_projection = np.concatenate((projection, padding), axis=1)

projection_fft = np.fft.fft(padded_projection)
# rf = np.transpose(_get_fourier_filter(projection_fft.shape[1], "shepp-logan"))
rf = ramp_filter_fancy(projection_fft.shape[1])
rf = apply_cutoff(rf, rf.max()/10)
plt.plot(rf.T)
plt.show()
projection_fft_filtered = rf * projection_fft

padded_projection_filtered = np.fft.ifft(projection_fft_filtered)
projection_filtered = padded_projection_filtered[:, 0:padded_projection.shape[1] // 2]

def compare_original2filtered(original, filtered, profile_y):
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.axhline(y=profile_y, ls="-", c="yellow")
    
    plt.subplot(2, 2, 3)
    plt.imshow(filtered)
    plt.axhline(y=profile_y, ls="-", c="yellow")
    
    plt.subplot(2, 2, 2)
    plt.plot(original[profile_y, :])

    plt.subplot(2, 2, 4)
    plt.plot(filtered[profile_y, :])
    plt.show()

compare_original2filtered(projection, np.real(projection_filtered), profile_y=1148)
