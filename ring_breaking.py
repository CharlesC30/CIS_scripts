import numpy as np
import matplotlib.pyplot as plt
import cv2
from imgutils import load_and_normalize
import derivatives as drv

def count_inner_contours(contour_hierarchy):
    hierarchy = np.squeeze(contour_hierarchy, axis=0)
    
    # this will be a 1D array with -1 when there is no parent, otherwise the index of the parent
    parents = hierarchy[:, -1]
    return np.count_nonzero(parents != -1)


image = load_and_normalize("/lhome/clarkcs/Cu-pins/pin-pore-examples/ict_pin_pore.npy", 16)
image = drv.imderiv_max(image, mode="center")

n_inner_contours = []
prev_binary_image = np.empty(image.shape)
for thresh in range(0, 4000):
    binary_image = image >= thresh

    contours, hierarchy = cv2.findContours(binary_image.astype(int), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(hierarchy)
    n_inner_contours.append(count_inner_contours(hierarchy))
    if len(n_inner_contours) > 1:
        if n_inner_contours[-2] != n_inner_contours[-1]:
            print(n_inner_contours[-2], n_inner_contours[-1])
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(prev_binary_image)
            ax1.set_title(f"threshold={thresh-1}")
            ax2.imshow(binary_image)
            ax2.set_title(f"threshold={thresh}")
            plt.show()
        
    prev_binary_image = binary_image.copy()






