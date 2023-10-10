import numpy as np
import h5py
import os

os.chdir("/lhome/clarkcs/cylinders/fromAmmar")

with h5py.File("projAtten-nonneg.hdf5", "w") as dest_file:
    with h5py.File("projAtten.hdf5", "r") as src_file:
        for name in src_file.keys():
            if name != "Image":
                src_file.copy(src_file[name], dest_file, name)
        projections = np.array(src_file["Image"])
        projections[projections < 0] = 0
        dest_file.create_dataset("Image", data=projections)
        