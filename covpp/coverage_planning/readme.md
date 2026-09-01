# Coverage Planning Module

Generates coverage paths for each cell.

## Components

- **CoveragePPHandler** (`coverage_pp_hl.py`): Loads the configured coverage planner
- **CoveragePathPlannerBase** (`coverage_pp_base.py`): Base class for all coverage planners
- **CoveragePathHighLevel** (`coverage_pp_hl.py`): Orchestrates multi-cell coverage planning

## Available Methods

- **boust**: Back-and-forth parallel sweeping (see [boustrophedon/readme.md](boustrophedon/readme.md))
- **contour**: Inward spiraling boundary-following (see [contour/readme.md](contour/readme.md))
- **grid_energy_min**: Grid-based greedy nearest-neighbor (see [local_energy_minimization/readme.md](local_energy_minimization/readme.md))

## Data Flow

```
PCell (cell with geometry)
  ↓
CoveragePPHandler (loads plugin)
  ↓
CoveragePlannerPlugin.coverage_planning()
  ↓
MyPath (ordered waypoints in meters)
  ↓
SequenceHandler (connect cells)
```

## Configuration

Select in `configs.py`:

```python
pipeline_config.covpp_method = "boust"  # or "contour", "grid_energy_min"
```

## Base Class Features

All coverage planners inherit from `CoveragePathPlannerBase`, providing:

- Cost map generation (inflated obstacle maps)
- Endpoint validation
- Coordinate transformation (local ↔ global)
- Path filtering and rotation
```

Paths are in both pixel and meter coordinates for flexibility.
