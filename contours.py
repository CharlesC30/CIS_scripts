import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
import cv2

image = np.load("/lhome/clarkcs/Cu-pins/high_quality/three_pins/one-pin-pore_sobel_8bit.npy")

thresh = 90
binary_image = (image >= thresh).astype(int)
skeleton = morphology.skeletonize(binary_image)
contours = measure.find_contours(skeleton)
plt.imshow(skeleton)
# for cnt in contours:
#     plt.plot(cnt[:, 1], cnt[:, 0], linewidth=1)
plt.show()

# countours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# print(countours[0].shape, countours[1].shape)
# plt.imshow(binary_image)
# for c in np.squeeze(countours[0]):
#     print(c)
#     plt.plot(c.T)
# for c in countours[1]:
#     plt.plot(c)

# plt.show()

# for thresh in range(0, 256):
#     binary_image = image >= thresh
#     contours = measure.find_contours(binary_image)
#     fig, ax = plt.subplots()
#     ax.imshow(binary_image)
#     for contour in contours:
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     plt.show()
