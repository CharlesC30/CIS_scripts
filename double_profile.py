import numpy as np
import matplotlib.pyplot as plt

slice_paths = [
    "/lhome/clarkcs/cylinders/fromAmmar/N_projs/reco-sparse2-unf",
    "/lhome/clarkcs/cylinders/fromAmmar/N_projs/reco-sparse3-unf",
    "/import/scratch/tmp-ct-3/charles_tmp/N_projs/reco-sparse4-unf",
    "/import/scratch/tmp-ct-3/charles_tmp/N_projs/reco-sparse5-unf",
    "/import/scratch/tmp-ct-3/charles_tmp/N_projs/reco-sparse8-unf",
    "/import/scratch/tmp-ct-3/charles_tmp/N_projs/reco-sparse16-unf",
]

profile_ys = [500, 1500]

for path in slice_paths:
    slice1148 = np.load(path + "/slice1148.npy")
    plt.subplot(1, 2, 1)
    plt.imshow(slice1148)
    
    for y in profile_ys:
        plt.axhline(y=y, ls='-', c="yellow")
    plt.subplot(1, 2, 2)
    for y in profile_ys:
        plt.plot(slice1148[y, :])

    plt.show()