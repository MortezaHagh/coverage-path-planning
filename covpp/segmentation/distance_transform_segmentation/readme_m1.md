# Distance Transform Segmentation

## Main idea

This segmentor treats the free workspace as a geometric field rather than a
binary pixel mask.

The key idea is that the distance transform encodes how far each free pixel is
from the nearest obstacle. Local maxima of that field tend to lie in the most
open parts of a room, so they are good seeds for segmentation.

The method therefore:

1. computes the distance-to-obstacle map,
2. smooths it slightly to suppress noise and merge nearby peaks,
3. detects room-center candidates as local maxima,
4. turns those peaks into markers,
5. expands the markers with watershed on the negative distance field, and
6. merges tiny regions that are likely artifacts.

In algorithmic terms, this is marker-controlled watershed segmentation on a
distance surface.

## Method

```
Occupancy Map
      │
      ▼
Distance Transform
      │
      ▼
Gaussian Blur (optional)
      │
      ▼
Local Maxima Detection
      │
      ▼
Marker Image
      │
      ▼
Marker-Controlled Watershed
      │
      ▼
Merge Small Regions
```


## Inputs

Binary occupancy map where free space is non-zero and obstacles are zero.

The implementation expects the workspace map from `ZoneMapData` and uses the
free-space mask as the watershed domain.

## Algorithmic View

The distance transform converts each free pixel into a clearance value. Pixels
with high clearance are preferred seeds because they are far from walls and are
less likely to sit on narrow artifacts. After smoothing, local maxima are
extracted with a minimum separation and minimum peak height so that one room
usually produces one seed.

Watershed then grows each marker until competing basins meet. Since the input
surface is `-distance`, the watershed boundaries fall on ridges of the distance
field, which usually correspond to wall-constrained necks, doorways, or other
narrow transitions between open areas.

After expansion, tiny regions are merged into the most common neighboring label
to remove spurious basins created by noise, discretization, or over-sensitive
peak detection.

## Pseudocode

FUNCTION DistanceTransformSegmentation(binary_map):

    distance ← DistanceTransform(binary_map)
    distance ← GaussianBlur(distance)

    peaks ← LocalMaxima(distance,
                       min_distance,
                       min_height)

    markers ← CreateMarkerImage(peaks)

    labels ← Watershed(-distance,
                       markers,
                       mask=binary_map)

    labels ← MergeSmallRegions(labels)

    RETURN labels

## Parameter Meaning

- `sigma`: smoothing radius for the distance field before peak detection.
- `min_peak_distance`: minimum spacing between candidate room centers.
- `min_peak_value`: minimum clearance required for a peak to be accepted.
- `min_room_area`: smallest region that should survive post-processing.

## Complexity

Distance transform: `O(N)`

Gaussian smoothing: approximately `O(N)`

Peak detection: depends on the peak window and image size, typically close to `O(N)` in practice

Watershed: `O(N)` to `O(N log N)` depending on the implementation details

Region merging: `O(N)`

`N` is the number of pixels in the workspace map.

## Output

- `labels_map`: integer segmentation map with one label per region.
- `polygons`: Shapely polygons built from the label contours.
- `adjacency_matrix`: region adjacency inferred from neighboring labels near
  the watershed boundaries.
- `map_ws_segmented`: visualization map with segmentation lines drawn in.

## Notes

This approach is usually better than a plain BFS growth step when the goal is to
place boundaries in narrow geometric transitions instead of simply spreading
labels uniformly. It is especially useful when the workspace has rooms with
distinct clearance peaks and constricted connectors between them.