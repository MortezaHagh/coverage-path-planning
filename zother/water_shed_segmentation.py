""" Watershed segmentation example."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max  # pylint: disable=E0611

# Load image from path
# im_path = "/home/morteza/Documents/00-CovPP/covpp/results/map_fr52/AMain/merge.png"
IM_PATH = "/home/morteza/Documents/00-CovPP/covpp/results/results_r_0.3/map_fr52/District/maps/local_map_3_cleaning.png"
image = plt.imread(IM_PATH)

# make sure the image is binary
if image.ndim == 3:
    image = np.mean(image, axis=2)  # Convert to grayscale by averaging channels
image = image > 0.5  # Threshold to create a binary image

# # Generate an initial image with two overlapping circles
# x, y = np.indices((80, 80))
# x1, y1, x2, y2 = 28, 28, 44, 52
# r1, r2 = 16, 20
# mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
# mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
# image = np.logical_or(mask_circle1, mask_circle2)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True, constrained_layout=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

# plot coords on the last image
ax[2].plot(coords[:, 1], coords[:, 0], 'r.', markersize=5, label='Local maxima')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
