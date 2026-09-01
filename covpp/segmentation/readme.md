# Segmentation Module

Splits each zone into planning cells.

## Components

- **SegmentationHandler** (`segmentation_handler.py`): Loads the configured segmentation plugin
- **SegmentationBase** (`segmentation_base.py`): Base class for all segmentation methods
- **CellBase** (`cell_base.py`): Geometric cell (polygon)
- **PCell** (`cell_planning.py`): Planning-ready cell with a cropped binary map, optional rectangle analysis, and coverage-direction metadata
- **MapPolygons** (`map_polygons.py`): Workspace boundary and obstacles

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

## Configuration

Select in `configs.py`:

```python
pipeline_config.segmentation_method = "voronoi"  # simple, skeleton, morphological, distance_tf, sweep_skeleton
```
