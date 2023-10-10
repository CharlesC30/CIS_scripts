import numpy as np
import h5py
import matplotlib.pyplot as plt

vol_paths = ["/import/scratch/tmp-ct-3/AM/AluCylinder2/CtutilRealReco/volume.hdf5", 
             "/lhome/clarkcs/cylinders/fromAmmar/reco/volume.hdf5"]
slice_n =  289
profile_y = 1148

for i, path in enumerate(vol_paths):
    with h5py.File(path, "r") as file:
        print(path)
        vol_array = np.array(file["Volume"])
    img_array = vol_array[:, slice_n, :]
    profile_line = img_array[profile_y, :]
        
    plt.subplot(1, len(vol_paths)+1, i+1)
    plt.imshow(img_array, "gray")
    plt.axhline(y=profile_y, ls="-", c="yellow")

    ax = plt.subplot(1, len(vol_paths)+1, len(vol_paths)+1)
    # ax = plt.axes()
    ax.plot(profile_line)

plt.show()

    