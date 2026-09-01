# Contour Coverage Planning

Inward spiraling path that follows cell boundaries and internal features.

## Algorithm

1. Extract cell boundary and obstacle polygons
2. Create offset contours at specified spacing inward from boundary
3. For each obstacle inside cell, create offset contours around it
4. Order contours to form connected spiral path
5. Generate waypoints along each contour
6. Connect contours to minimize backtracking

## Input

- Cell polygon (exterior boundary and interior obstacles)
- Robot configuration
- Coverage spacing parameter

## Output

`MyPath` object containing:
- Waypoints following contour spiral inward
- Contour segments
- Path metrics

## Parameters

- **coverage_spacing**: Distance between concentric contours (meters)

## Configuration

Select in `PipeLineConfig`:

```python
pipeline_config.covpp_method = "contour"
pipeline_config.coverage_spacing = 0.5  # meters between contours
```

The contour method is often paired with segmentation methods that preserve cell shape information (e.g., `skeleton` or `voronoi` for irregular environments).

## Technical Notes

- Uses polygon offset operations (typically from Shapely library)
- Offset computation is relatively expensive; caches results when possible
- Robust handling of numerical precision issues in offset computation
- May subdivide large cells to improve path quality

## References

- Contour-based coverage planning
- Offset polygon computation
- Spiral traversal strategies in CPP
