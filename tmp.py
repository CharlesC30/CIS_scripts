# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Read the image (replace 'your_image.jpg' with your actual image file)
# # image = cv2.imread('pin_pore_data/one-pin-pore.npy', cv2.IMREAD_GRAYSCALE)
# image = np.load("pin_pore_data/one-pin-pore.npy")
# # Apply binary thresholding to create a binary mask
# _, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # Create a kernel for dilation
# kernel_size = 5
# kernel = np.ones((kernel_size, kernel_size), np.uint8)

# # Dilate the binary mask
# dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

# # Invert the binary mask to keep the original image within the outer edges
# inverted_mask = cv2.bitwise_not(binary_mask)

# # Bitwise AND the original image with the inverted mask to keep only the outer edges
# result = cv2.bitwise_and(image, image, mask=inverted_mask)

# # Bitwise OR the result with the dilated mask to add the dilated outer edges
# result_with_dilated_edges = cv2.bitwise_or(result, dilated_mask)

# # Display the original image, dilated outer edges, and the final result
# plt.figure(figsize=(10, 5))

# plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
# plt.subplot(132), plt.imshow(dilated_mask, cmap='gray'), plt.title('Dilated Mask')
# plt.subplot(133), plt.imshow(result_with_dilated_edges, cmap='gray'), plt.title('Result with Dilated Outer Edges')

# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Global variable to track zooming state
is_zooming = False

# Define a callback function to handle mouse clicks
def on_click(event):
    if event.button == 1 and event.inaxes and not is_zooming:  # Check if the left mouse button was clicked, inside the axes, and not zooming
        print(f'You clicked at ({event.xdata:.2f}, {event.ydata:.2f})')

# Define a callback function to handle zoom-related events
def on_zoom(event):
    global is_zooming
    is_zooming = True

# Define a callback function to handle zooming end
def on_release(event):
    global is_zooming
    is_zooming = False

# Generate some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot and axes object
fig, ax = plt.subplots()
ax.plot(x, y)

# Connect the callback functions to the appropriate events
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_zoom = fig.canvas.mpl_connect('scroll_event', on_zoom)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

# Show the plot
plt.show()