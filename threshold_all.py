import numpy as np
import matplotlib.pyplot as plt
import derivatives as drv
import cv2
import os

N_BITS = 16
max_val = 2 ** N_BITS

image = np.load("/lhome/clarkcs/Cu-pins/pin-pore-examples/ict_pin_pore.npy")
cv2.normalize(image, image, 0, max_val, cv2.NORM_MINMAX)
image = image.astype(int)

out_path = "/zhome/clarkcs/Pictures/thresholding/Xingyu_pore_test_16bit"
if not os.path.exists(out_path):
    os.mkdir(out_path)
os.chdir(out_path)

# image_derivative = drv.imderiv_max(image, mode="center")
# center_derivative = drv.imderiv_max(image, mode="center")
image_derivative = drv.full_sobel(image)

plt.gray()
for thresh in range(0, max_val):
    binary_image = image >= thresh
    binary_image_derivative = image_derivative >= thresh
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"threshold={thresh}")
    ax1.imshow(binary_image)
    ax2.imshow(binary_image_derivative)
    plt.savefig(f"thresh_{thresh}.png")
    plt.close(fig)


