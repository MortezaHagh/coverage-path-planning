# Sweep-Skeleton Decomposition

Hybrid segmentation combining sweeping patterns with skeleton-based structure detection.

<!-- todo doc -->

## Algorithm

1. Compute distance transform and skeleton
2. Generate sweep lines in primary direction
3. Identify intersections and sweep-based boundaries
4. Combine skeleton junctions with sweep separations
5. Grow cells from seed points using combined signals
6. Perform wavefront expansion to cover entire space

## Input

- Binary occupancy map

## Output

List of `CellBase` objects with balanced, structured cell decomposition.

## Parameters

- **primary_sweep_direction**: Angle of sweep pattern (degrees)
- **sweep_spacing**: Distance between sweep lines (meters)
- **skeleton_weight**: Influence of skeleton vs. sweep signals (0.0-1.0)
- **min_cell_area**: Minimum acceptable cell area (m²)

## Configuration

Select in `PipeLineConfig`:

```python
pipeline_config.segmentation_method = "sweep_skeleton"
pipeline_config.sweep_direction = 0
pipeline_config.sweep_spacing = 2.0
pipeline_config.skeleton_weight = 0.5
```

## Configuration

In `PipeLineConfig`:

```python
pipeline_config.segmentation_method = "sweep_skeleton"
# Additional parameters in segmentation config:
sweep_direction = 0  # 0° (horizontal), 90° (vertical)
sweep_spacing = 2.0  # meters between sweep lines
skeleton_weight = 0.5  # 0.0=sweep-only, 1.0=skeleton-only
```

## Comparison with Other Methods

| Method | Balance | Structure | Speed | Complexity |
|--------|---------|-----------|-------|-----------|
| Simple | Perfect | None | Fastest | 0 |
| Voronoi | Good | Balanced | Fast | 1 |
| Skeleton | Fair | Excellent | Slow | 3 |
| Distance Transform | Fair | Accessibility | Medium | 2 |
| Sweep-Skeleton | Excellent | Excellent | Slow | 4 |

## Parameter Tuning Guide

- **Sweep direction**: 0° for horizontal rooms, 90° for vertical, 45° for diagonal bias
- **Sweep spacing**: Larger = fewer cells, smaller = more detailed decomposition
- **Skeleton weight**: Higher = more skeleton influence, lower = more sweep influence
- **Min cell area**: Filter noise, typically 0.1-1.0 m²

## References

- Hybrid decomposition strategies
- Sweep-based and skeleton-based methods
- Adaptive space partitioning
