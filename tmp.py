import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image (replace 'your_image.jpg' with your actual image file)
# image = cv2.imread('pin_pore_data/one-pin-pore.npy', cv2.IMREAD_GRAYSCALE)
image = np.load("pin_pore_data/one-pin-pore.npy")
# Apply binary thresholding to create a binary mask
_, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Create a kernel for dilation
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Dilate the binary mask
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

# Invert the binary mask to keep the original image within the outer edges
inverted_mask = cv2.bitwise_not(binary_mask)

# Bitwise AND the original image with the inverted mask to keep only the outer edges
result = cv2.bitwise_and(image, image, mask=inverted_mask)

# Bitwise OR the result with the dilated mask to add the dilated outer edges
result_with_dilated_edges = cv2.bitwise_or(result, dilated_mask)

# Display the original image, dilated outer edges, and the final result
plt.figure(figsize=(10, 5))

plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(dilated_mask, cmap='gray'), plt.title('Dilated Mask')
plt.subplot(133), plt.imshow(result_with_dilated_edges, cmap='gray'), plt.title('Result with Dilated Outer Edges')

plt.show()
