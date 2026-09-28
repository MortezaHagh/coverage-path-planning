# Skeleton-Based Segmentation

Skeleton-based segmentation decomposes a zone using the medial axis (skeleton) of the free space.

<!-- todo doc -->

## Algorithm

1. Compute distance transform of free space (distance to nearest obstacle)
2. Apply medial axis/skeleton extraction (thinning algorithm)
3. Identify skeleton endpoints and junction points (branching)
4. Decompose space using skeleton structure:
   - Skeleton branches become cell separators
   - Junction points define cell boundaries
   - Regions between branches become cells
5. Grow cells from skeleton using wavefront expansion
6. Convert to polygon representation for coverage planning

## Input

- Binary occupancy map (1 = free, 0 = obstacle)

## Output

List of `CellBase` objects representing cell polygons with preserved connectivity.

## Parameters

- **Min region size**: Filters small isolated cells
- **Skeleton pruning**: Removes short branches
- **Dilation**: Smooths skeleton for cleaner decomposition

## Configuration

Select in `PipeLineConfig`:

```python
pipeline_config.segmentation_method = "skeleton"
```
| Voronoi | Balanced | Fast | Medium | General |
| Skeleton | Topology | Slow | Adaptive | Complex spaces |
| Distance Transform | Accessibility | Medium | Adaptive | Natural clusters |
| Morphological | Erosion | Medium | Adaptive | Chamber detection |

