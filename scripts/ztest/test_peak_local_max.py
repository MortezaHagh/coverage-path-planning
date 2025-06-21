from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
# from skimage import data, img_as_float

# Load image from path
im_path = "/home/morteza/Documents/00-CPP/00-CPP/Data/map_fr52/AMain/merge.png"
image = plt.imread(im_path)

# apply distance transform
image = ndi.distance_transform_edt(image)
# Normalize the image to the range [0, 1]
# image = image / image.max()
# display the image
# plt.imshow(image, cmap=plt.cm.gray)

# image_max is the dilation of image with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(image, size=1, mode='constant')

print("maximum value in image_max:", image_max.max())
print("minimum value in image_max:", image_max.min())

# Comparison between image_max and image to find the coordinates of local maxima
# coordinates = peak_local_max(image, min_distance=1, threshold_abs=10)
# get coordinates of local maxima in image_max with a minimum distance smaller than 20
coordinates = peak_local_max(image, min_distance=1, threshold_abs=1)

# display results
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(image, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()

plt.show()
