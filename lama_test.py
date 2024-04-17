from simple_lama_inpainting import SimpleLama
import numpy as np
import matplotlib.pyplot as plt

simple_lama = SimpleLama()

img_path = "/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_10projs_center-slice_3ch.npy"
mask_path = "/lhome/clarkcs/aRTist_simulations/cylinder/10projs_mask.npy"

image = np.load(img_path)
mask = np.load(mask_path)

result = simple_lama(image, mask)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

ax1.imshow(image[:, :, 0])
ax2.imshow(mask)
ax3.imshow(image[:, :, 0] * (mask - 255))
red, green, blue = result.split()
ax4.imshow(red)
plt.show()

