import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from imgutils import load_and_normalize
import derivatives as drv

image = load_and_normalize("/lhome/clarkcs/Cu-pins/pin-pore-examples/ict_pin_pore.npy", 16)
image = drv.imderiv_max(image, mode="center")
ccl_count = []

prev_n_labels = 0
prev_binary_image = np.empty(image.shape)
for thresh in range(0, 4000):
    binary_image = image < thresh
    labels, n_labels = measure.label(binary_image, return_num=True, connectivity=1)

    ccl_count.append(n_labels)

    if n_labels != prev_n_labels and thresh != 0:
        if n_labels < 1_000_000:
            print(prev_n_labels, n_labels)
            plt.subplot(1, 2, 1)
            plt.title(f"threshold={thresh-1}")
            plt.imshow(prev_binary_image)
            plt.subplot(1, 2, 2)
            plt.title(f"threshold={thresh}")
            plt.imshow(binary_image)
            plt.show()

    prev_n_labels = n_labels
    prev_binary_image = binary_image.copy()

plt.plot(ccl_count)
plt.show()
