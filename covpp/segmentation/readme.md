# Segmentation Module

Splits each zone into planning cells.

## Components

- **SegmentationHandler** (`segmentation_handler.py`): Loads the configured segmentation plugin
- **SegmentationBase** (`segmentation_base.py`): Base class for all segmentation methods
- **CellBase** (`cell_base.py`): Geometric cell (polygon)
- **CellMetrics** (`cell_base.py`): Per-cell geometric metrics calculated during cell creation
- **PCell** (`cell_planning.py`): Planning-ready cell with a cropped binary map, optional rectangle analysis, and coverage-direction metadata
- **MapPolygons** (`map_polygons.py`): Workspace boundary and obstacles
- **SegmentationMetrics** (`segmentation_metrics.py`): Average metrics calculated across the processed cells

## Available Methods

- **simple**: Single cell per zone
- **voronoi**: Voronoi partitioning
- **skeleton**: Skeleton-based structure-aware decomposition (see [skeleton_segmentation/readme.md](skeleton_segmentation/readme.md))
- **distance_tf**: Threshold-based accessibility clustering (see [distance_transform_segmentation/readme.md](distance_transform_segmentation/readme.md))
- **morphological**: Erosion-based room detection (see [morphological_segmentation/readme.md](morphological_segmentation/readme.md))
- **sweep_skeleton**: Sweep-based skeleton decomposition (see [sweep_skeleton_decomposition/readme.md](sweep_skeleton_decomposition/readme.md))

## Data Flow

```
ZoneMapData
  ↓
SegmentationHandler (loads plugin)
  ↓
SegmentationPlugin.run() (standard lifecycle)
  ↓
CellBase list (bcells)
PCell objects (with planning info)
  ↓
CoveragePPHandler (one path per cell)
```

## Cell Metrics

`CellBase` stores geometry in global pixel coordinates. Its `metric` field is a
`CellMetrics` instance containing:

| Metric | Meaning | Units |
| --- | --- | --- |
| `area` | Polygon area | square pixels |
| `perimeter` | Polygon boundary length | pixels |
| `free_perimeter` | Number of sampled boundary pixels that are free in the binary workspace map | pixels |
| `compactness_a` | `area / perimeter**2` | dimensionless |
| `compactness_b` | `area / rotated_bounding_box_area` | dimensionless |

`free_perimeter` is calculated after segmentation by sampling the one-pixel
expanded cell boundary against the workspace map (`255` is free space and `0`
is occupied space). The map resolution is not applied automatically to these
values. Convert area and length to physical units with the zone resolution:
`area_m2 = area * resolution**2` and `length_m = length * resolution`.

Each cell is also classified for planning:

- `size_type` is set to `small` when `area < 12 * pi * robot_radius_px**2`.
- `slim_type` is set to `slim` when the maximum inscribed-circle radius is less than `2 * robot_radius_px`.
- `do_planning` is set to `False` when `area < 4 * pi * robot_radius_px**2`.

Here, `robot_radius_px` is derived from the configured cleaning radius and map
resolution. A cell can have both `small` and `slim` classifications.

## Segmentation Metrics

After cells are merged and converted to `PCell` objects,
`SegmentationHandler.metrics` contains a `SegmentationMetrics` instance with
the following arithmetic means across all cells:

- `average_area`
- `average_perimeter`
- `average_free_perimeter`
- `average_compactness_a`
- `average_compactness_b`

The first three retain the pixel-based units described above; both compactness
averages are dimensionless. Metrics are computed after the configured
segmentor runs, any zero-cell fallback to simple segmentation occurs, and
eligible cells are merged.

## Configuration

Select in `configs.py`:

```python
pipeline_config.segmentation_method = "voronoi"  # simple, skeleton, morphological, distance_tf, sweep_skeleton
```
