from simple_lama_inpainting import SimpleLama
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

simple_lama = SimpleLama()

sparse_view_path = "/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_10projs_center-slice.npy"
full_view_path = "/lhome/clarkcs/aRTist_simulations/cylinder/1mm-cylinder_1000projs_center-slice.npy"
mask_path = "/lhome/clarkcs/aRTist_simulations/cylinder/10projs_mask.npy"
save_path = "/lhome/clarkcs/aRTist_simulations/cylinder/lama_inpainting"

sparse_view = np.load(sparse_view_path)
full_view = np.load(full_view_path)
mask = np.load(mask_path)
# mask = full_view.copy()
# thresh = threshold_otsu(full_view)
# mask[mask >= thresh] = 255
# mask[mask < thresh] = 0
# mask = mask.astype(np.uint8)

# sparse_view_pad = np.stack([sparse_view, np.zeros(sparse_view.shape), np.zeros(sparse_view.shape)], axis=2)
artifacts = sparse_view.astype(np.float64) - full_view.astype(np.float64)
artifacts_3ch = np.stack(3 * [artifacts], axis=2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(sparse_view)
ax2.imshow(full_view)
ax3.imshow(artifacts)
ax1.set_title("sparse view ($N_p$=10)")
ax2.set_title("full view ($N_p$=1000)")
ax3.set_title("artifacts = sparse view - full view")
plt.show()

print(artifacts_3ch.shape, mask.shape)
result = simple_lama(artifacts_3ch, mask)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(artifacts_3ch[:, :, 0])
ax2.imshow(mask)
ax3.imshow((mask - 255) * artifacts_3ch[:, :, 0])
result = np.array(result)
red = result[:, :, 0]
green = result[:, :, 1]
blue = result[:, :, 2]
ax4.imshow(red)
ax1.set_title("artifacts (ground truth)")
ax2.set_title("binary mask")
ax3.set_title("masked artifacts")
ax4.set_title("NN prediction")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(red)
ax2.imshow(green)
ax3.imshow(blue)
plt.show()

np.save(save_path + "/nn_prediction", red)
np.save(save_path + "/ground_truth", artifacts)
