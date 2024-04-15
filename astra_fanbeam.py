import astra
import numpy as np
import matplotlib.pyplot as plt


# Create phantom image
phantom_size = 256
phantom = np.zeros((phantom_size, phantom_size))
phantom[phantom_size//4:3*phantom_size//4, phantom_size//4:3*phantom_size//4] = 1

# Create geometry for fanbeam projection
detectors_count = 360  # Number of detectors
detector_size = 1.0  # Size of each detector cell
angles = np.linspace(0, np.pi, detectors_count, endpoint=False)  # Angles for projection

# Create projector configuration
source_object = 512
object_detector = 512
proj_geom = astra.create_proj_geom('fanflat', detector_size, detectors_count, angles, source_object, object_detector)

# Create volume geometry (in this case, a simple square)
vol_geom = astra.create_vol_geom(phantom_size, phantom_size)

# Create phantom data buffer
vol_id = astra.data2d.create('-vol', vol_geom, phantom)

# Create projector
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

# FP to create the sinogram
sino_id, sino_data = astra.create_sino(vol_id, proj_id)

# Display or further process projection_data
plt.imshow(sino_data)
plt.show()

# Cleanup
astra.data2d.delete(sino_id)
astra.data2d.delete(vol_id)
astra.projector.delete(proj_id)

