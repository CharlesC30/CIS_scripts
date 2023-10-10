import h5py
import numpy as np
import os

n = 992

volume_dirs = [
    "/lhome/clarkcs/75x15_9/rawdata-sparse140-recon-unf",
]

for dir in volume_dirs:
    os.chdir(dir)
    with h5py.File("volume.hdf5") as file:
        slice_n = np.array(file["Volume"][:, n, :])
        np.save(f"slice{n}", slice_n)
