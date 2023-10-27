import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, filters
from ccl_2d_thresh import ccl_threshold
import derivatives as drv

image = np.load("/lhome/clarkcs/Cu-pins/pin-pore-examples/ict_pin_pore.npy")
image[image < 0] = 0
cv2.normalize(image, image, 0, 256, cv2.NORM_MINMAX)
image = image.astype(int)

thresh = ccl_threshold(image, plot_results=True)
# thresh = filters.threshold_otsu(image)
print(thresh)
binary_image = image >= thresh

plt.imshow(binary_image)
plt.show()

