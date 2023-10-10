import numpy as np
import h5py
import os

os.chdir("/lhome/clarkcs/75x15_9")

with h5py.File("rawdata-sparse70.hdf5", "w") as dest_file:
    with h5py.File("rawdata.hdf5", "r") as src_file:
        for name in src_file.keys():
            if name != "Angle" and name != "Image":
                src_file.copy(src_file[name], dest_file, name)
        dest_file["Angle"] = src_file["Angle"][::70]
        dest_file["Image"] = src_file["Image"][::70]
