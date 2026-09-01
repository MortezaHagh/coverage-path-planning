# Coverage Path Planning Documentation

## Summary

CovPP is a plugin-driven coverage path planning pipeline for occupancy-grid maps. It takes a map, prepares a workspace, creates a district, splits each zone into cells, plans coverage paths, sequences them, and evaluates the final result.

The pipeline is controlled by the entry-point groups defined in `pyproject.toml`:

- `segmentors`
- `coverage_planners`
- `path_planners`
- `sequencers`

## End-to-End Flow

```text
InputsProcessor -> MapImageProcessor -> DistrictGen -> SpaceAnalyzer
  -> DistrictData -> ZoneData
    -> SegmentationHandler -> CoveragePPHandler -> SequenceHandler -> ZoneEvaluationHandler
```

At the district level, `DistrictData` collects the zone results and writes the merged planning map and aggregated statistics.

## Main Modules

### Base Layer

- `covpp/base/configs.py`: pipeline and robot configuration dataclasses
- `covpp/base/geometry.py`: point and path helpers
- `covpp/base/image_processor.py`: map preprocessing
- `covpp/base/inputs_processor.py`: input loading and validation
- `covpp/base/logger.py`: project logger

### District and Space Analysis

- `covpp/district/create_district.py`: district selection and rectangle creation
- `covpp/district/district.py`: district orchestration and zone creation
- `covpp/district/zone.py`: zone pipeline controller
- `covpp/district/zone_map_data.py`: zone-specific data container
- `covpp/spa/space_analyzer.py`: workspace selection and free-space analysis

### Segmentation

- `covpp/segmentation/segmentation_handler.py`: loads the configured segmentation plugin
- `covpp/segmentation/segmentation_base.py`: base class for segmentation methods
- `covpp/segmentation/simple_segmentation/`: simple segmentation plugin
- `covpp/segmentation/voronoi_segmentation/`: Voronoi-based segmentation plugin
- `covpp/segmentation/skeleton_segmentation/`: skeleton-based segmentation plugin

### Coverage Planning

- `covpp/coverage_planning/coverage_pp_hl.py`: loads the configured coverage planner
- `covpp/coverage_planning/coverage_pp_base.py`: base class for cell-level coverage planners
- `covpp/coverage_planning/boustrophedon/`: boustrophedon planner
- `covpp/coverage_planning/contour/`: contour planner
- `covpp/coverage_planning/local_energy_minimization/`: grid and local-energy planner

### Path Planning and Sequencing

- `covpp/path_planning/path_planner_base.py`: shared low-level path planning logic
- `covpp/path_planning/path_planner_pyastar.py`: A* implementation
- `covpp/sequence/sequence_handler.py`: sequencing and path connection controller
- `covpp/sequence/sequence_optimizer_base.py`: sequencing base class
- `covpp/sequence/dfs_optimizer/`: DFS optimizer
- `covpp/sequence/milp_optimizer/`: MILP optimizer
- `covpp/sequence/simple_optimizer/`: legacy simple optimizer, not registered by default

### Evaluation

- `covpp/evaluation/zone_evaluation_handler.py`: zone-level evaluation entry point
- `covpp/evaluation/zone_evaluation_base.py`: zone stats and evaluator base classes
- `covpp/evaluation/district_evaluation.py`: district-level aggregation

## Configuration

The main configuration objects live in `covpp/base/configs.py`.

- `PipeLineConfig`: selects segmentation, coverage planning, path planning, and sequencing methods
- `MapConfig`: map directories and metadata
- `SpaceConfig`: workspace safety distances and cleaning radius
- `PlanningConfig`: waypoint spacing and cleaning parameters
- `RobotConfig`: robot motion assumptions

The full method name is built from the selected segmentation, coverage, and sequencing methods. That name is used in output paths and file names.

## Entry Points

| Group | Current Plugins |
| --- | --- |
| `segmentors` | `simple`, `voronoi`, `skeleton`, `distance_transform`, `morphological` |
| `coverage_planners` | `boust`, `contour`, `grid_energy_min` |
| `path_planners` | `astar` |
| `sequencers` | `dfs`, `milp`, `simple` |

## Algorithm Documentation

Detailed algorithm descriptions and pseudocode are provided in the module-specific README files:

### Segmentation Methods

- **Simple**: Single-cell zone decomposition
- **Voronoi**: Balanced spatial partitioning
- **Skeleton**: Structure-aware medial axis decomposition (see [skeleton_segmentation/readme.md](../covpp/segmentation/skeleton_segmentation/readme.md))
- **Distance Transform**: Accessibility-based thresholding (see [distance_transform_segmentation/readme.md](../covpp/segmentation/distance_transform_segmentation/readme.md))
- **Morphological**: Erosion-based room detection (see [morphological_segmentation/readme.md](../covpp/segmentation/morphological_segmentation/readme.md))

### Coverage Planning Methods

- **Boustrophedon**: Back-and-forth parallel sweeping (see [boustrophedon/readme.md](../covpp/coverage_planning/boustrophedon/readme.md))
- **Contour**: Inward spiraling boundary-following (see [contour/readme.md](../covpp/coverage_planning/contour/readme.md))
- **Grid Energy Minimization**: Grid-based greedy nearest-neighbor (see [local_energy_minimization/readme.md](../covpp/coverage_planning/local_energy_minimization/readme.md))

### Sequencing Methods

- **DFS**: Depth-first traversal heuristic
- **MILP**: Mixed Integer Linear Programming optimization (via OR-Tools)
- **Simple**: Nearest-neighbor greedy approach

Additional algorithm details including pseudocode, complexity analysis, and usage recommendations are available in [module_details.md](module_details.md).

## Output Layout

Planning results are written to the `maps/` tree.

```text
maps/<map_name>/
  AMain/
  District/
    result.json
    mp_<method>.png
    Zone_<i>/
      segmentation/
      <method>/
      result.json
```

Zone result files are serialized from `ZoneStatsBase`. District-level metrics are aggregated from zone statistics and saved by `DistrictEvaluationHandler`.

## Adding New Methods

To add a new algorithm, implement the matching base class, register it in `pyproject.toml`, and select it through `PipeLineConfig`.

- New segmentation: subclass `SegmentationBase` and register under `segmentors`
- New coverage planner: subclass `CoveragePathPlannerBase` and register under `coverage_planners`
- New path planner: subclass `PathPlannerBase` and register under `path_planners`
- New sequencing method: subclass `SeqOptimizerBase` and register under `sequencers`

Keep plugin names short, stable, and consistent with the method names used in configuration.

- **`ZoneEvaluationHandler`**:

  - Loads evaluation strategy based on requirements
  - Computes metrics using standardized interface

- **`SequenceHandler`**:
  - Selects optimization algorithm from config
  - Creates cell traversal order using pluggable optimizers

#### 3. **Configuration-Driven Behavior**

All method selection is controlled through configuration files:

```python
# In configs.py
class PipeLineConfig:
    segmentation_method: str = "voronoi"   # or "simple", custom via entry points
    covpp_method: str = "boust"            # or "contour", custom via entry points
    pp_method: str = "astar"               # or any custom path-planner plugin name
    sequence_method: str = "simple"        # or "dfs", "milp", custom via entry points
```

This allows:

- **Runtime flexibility**: Change algorithms without code modification
- **Easy comparison**: Test different methods by changing config
- **Deployment options**: Different configs for different scenarios

#### 4. **Dynamic Strategy via Entry Points**

Each handler uses the Strategy pattern to select algorithms:

Internally we resolve plugins using `importlib.metadata.entry_points`:

```python
segmentors = entry_points(group="segmentors")
segmentation_method = PipeLineConfig().segmentation_method
if segmentation_method not in segmentors.names:
  raise ValueError("Segmentation method not recognized")
segmentor_cls = segmentors[segmentation_method].load()
self.segmentor = segmentor_cls(self.zmd, self.map_polygons)
self.segmentor.run()
```

#### 5. **Data Transfer Objects (DTO)**

Standardized data structures ensure compatibility:

- `MapData`, `ZoneMapData`: Map and workspace information
- `CellBase`, `PCell`: Cell representation
- `MyPoint`, `MyPath`: Geometric primitives
- `ZoneStatsBase`, `DistrictStatsBase`: Statistics containers

These DTOs allow different implementations to communicate seamlessly.

## Adding New Methods (Plugin Development)

For detailed instructions on how to add new methods, please refer to [Plugin Development](plugin_development.md).

## End-to-End Pattern

Sequential processing stages with well-defined interfaces:

```text
Input → Process → Segment → Plan → Sequence → Evaluate
```

Each stage:

- Receives standardized input (DTOs)
- Performs specific transformation
- Produces standardized output
- Can be tested independently

## Dependencies

- **NumPy**: Numerical operations
- **OpenCV (cv2)**: Image processing
- **Matplotlib**: Visualization
- **Shapely**: Geometric operations
- **OR-Tools**: Optimization (MILP for sequencing)
- **PyYAML**: Configuration loading
- **Colorama**: Colored logging

## Usage Examples

```python
# Command line
covpp Map2
# or
python -m covpp.main Map2
```

## References

- Boustrophedon decomposition: Classical coverage path planning
- Voronoi decomposition: Optimal space partitioning
- Distance Transform: Contour-based path generation
- Plugin Architecture: Extensible software design patterns

## Notes on Map Layers

- `ws`: workspace (feasible robot-center positions)
- `cleaning`: cleanable area given the cleaning radius
- `sf`: selected free-space island (single connected component)
- `orig`: original (or eroded-back) workspace for context
- `map_planning`/`map_segmentation`: visualization overlays

## Entry Point Groups

- `segmentors`: segmentation methods
- `coverage_planners`: coverage path planners
- `path_planners`: low-level point-to-point planners
- `sequencers`: sequence optimizers
