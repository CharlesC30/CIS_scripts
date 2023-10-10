import numpy as np
import matplotlib.pyplot as plt
import skimage

slice_paths = [
    "/lhome/clarkcs/75x15_9/rawdata-sparse140-recon-unf/slice992.npy", 
    "/lhome/clarkcs/75x15_9/rawdata-sparse70-recon-unf/slice992.npy", 
    "/lhome/clarkcs/75x15_9/rawdata-recon-unf/slice992.npy", 
    ]

fig, axs = plt.subplots(1, len(slice_paths)+1)

profile_y = [1000]

# for ax, path in zip(axs[:-1], slice_paths):
for ax, path in zip(axs, slice_paths):
    for y in profile_y:
        slice = np.load(path)
        # slice = skimage.restoration.denoise_tv_chambolle(slice)
        # slice = skimage.filters.gaussian(slice, sigma=3)
        # thresh = skimage.filters.threshold_otsu(slice)
        # slice[slice < thresh] = 0
        ax.imshow(slice)
        ax.axhline(y=y, ls="-", c="yellow")
        axs[-1].plot(slice[y, :])

# makes the profile plot also square
asp = np.diff(axs[-1].get_xlim())[0] / np.diff(axs[-1].get_ylim())[0]
axs[-1].set_aspect(asp)

plt.show()
