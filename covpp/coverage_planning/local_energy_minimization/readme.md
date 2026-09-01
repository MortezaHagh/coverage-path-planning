# Local Energy Minimization Coverage Planning

Grid-based coverage path planner using greedy nearest-neighbor ordering.

## Algorithm

1. Create coverage grid with specified spacing inside cell
2. Mark grid points as valid (in free space) or invalid (in obstacles)
3. Initialize tour with arbitrary starting point
4. Use nearest-neighbor heuristic:
   - From current point, find closest unvisited valid grid point
   - Add to tour and move to that point
   - Repeat until all points visited
5. Optional: Apply local optimization (2-opt swaps)
6. Generate waypoints along the grid point path

## Input

- Cell polygon (boundary and obstacles)
- Robot configuration
- Grid and coverage spacing parameters

## Output

`MyPath` object containing:
- Waypoints in nearest-neighbor order
- Grid points used
- Path metrics

## Parameters

- **grid_spacing**: Size of coverage grid cells (meters)
- **coverage_spacing**: Minimum distance between coverage waypoints (meters)

## Configuration

Select in `PipeLineConfig`:

```python
pipeline_config.covpp_method = "grid_energy_min"
pipeline_config.grid_spacing = 0.5
pipeline_config.coverage_spacing = 0.3
```
