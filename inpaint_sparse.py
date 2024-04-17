from simple_lama_inpainting import SimpleLama
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

simple_lama = SimpleLama()

sparse_view_path = "/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_10projs_center-slice.npy"
full_view_path = "/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_1000projs_center-slice.npy"
# mask_path = "/lhome/clarkcs/aRTist_simulations/cylinder/10projs_mask.npy"

sparse_view = np.load(sparse_view_path)
full_view = np.load(full_view_path)
# mask = np.load(mask_path)
mask = full_view.copy()
thresh = threshold_otsu(full_view)
mask[mask >= thresh] = 255
mask[mask < thresh] = 0
mask = mask.astype(np.uint8)

sparse_view_pad = np.stack([sparse_view, np.zeros(sparse_view.shape), np.zeros(sparse_view.shape)], axis=2)
print(sparse_view_pad.shape, mask.shape)
result = simple_lama(sparse_view_pad, mask)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

ax1.imshow(sparse_view_pad[:, :, 0])
ax2.imshow(mask)
ax3.imshow((mask - 255) * sparse_view_pad[:, :, 0])
result = np.array(result)
red = result[:, :, 0]
green = result[:, :, 1]
blue = result[:, :, 2]
ax4.imshow(red + green + blue)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(red)
ax2.imshow(green)
ax3.imshow(blue)
plt.show()
