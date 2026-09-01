# Module Details

## 1. Base Module (`covpp/base/`)

Core utilities and data structures used throughout the pipeline.

### InputsProcessor

- Purpose: load input map files and configuration
- Key methods: `__init__(map_name)`, `set_paths()`, `load_yaml_file()`, `update_map_configs()`

### MapImageProcessor

- Purpose: process the grayscale map into the internal binary representation
- Key methods: `__init__(map_arr, map_config)`, `process_map()`, `set_map_data()`, `complete_map_data()`

### MapData

- Purpose: store the processed map and metadata
- Key attributes: `orig_map_gs`, `up_bin_map_u8`, `resolution`, `updated_w`, `updated_h`

### Configs (`configs.py`)

- `MapConfig`: map directories and metadata
- `PipeLineConfig`: active methods and pipeline switches
- `PlanningConfig`: waypoint spacing and planning parameters
- `SpaceConfig`: workspace safety distances and cleaning radius
- `RobotConfig`: robot motion assumptions

### Geometry (`geometry.py`)

- `MyPoint`: point container with x/y coordinates
- `MyPath`: ordered sequence of points

### Base utilities and logging

- `get_n_colors_cv()`, `get_n_colors_plt()`: color helpers
- `MyLogger`: project logger used by most handlers

## 2. GUI Module (`covpp/gui/`)

Interactive primitives for district selection and visualization.

### Rectangle (`rectangle.py`)

- Purpose: represent rotated rectangular regions
- Attributes: `xc`, `yc`, `w`, `h`, `rot`, `vertices_t`

### RecEventHandler (`rect_event_hl.py`)

- Purpose: interactive mouse and keyboard rectangle selection
- Used by `DistrictGen` when the district is not the whole map

### VisualRectangle (`visual_rectangle.py`)

- Purpose: drawing helpers for OpenCV canvases

## 3. Space Analysis Module (`covpp/spa/`)

Analyzes the selected district and builds workspace data.

### SpaceAnalyzer

- Purpose: convert the selected district into zone-ready map data
- Key methods: `create_local_map_ws_selection()`, `update_by_rectangle()`, `create_zone_map_data()`
- Key output: `zone_map_data`

### Rectangle helpers (`spa_rectangles.py`)

- `find_circumscribed_rectangle()`: enclosing rectangle helper
- `find_inscribed_rotated_rectangle()`: inscribed rectangle heuristic

## 4. District Module (`covpp/district/`)

Coordinates district creation, zone generation, and result aggregation.

### DistrictGen (`create_district.py`)

- Purpose: create the district rectangle
- Supports GUI selection or whole-map fallback

### DistrictData (`district.py`)

- Purpose: create zones, run planning, and merge district outputs
- Key methods: `create_zones_hl()`, `run_planning_pipeline()`, `update_map_planning()`, `save_map_planning()`

### ZoneData (`zone.py`)

- Purpose: run the full planning pipeline for one zone
- Pipeline: segmentation, coverage planning, sequencing, evaluation

### ZoneMapData (`zone_map_data.py`)

- Purpose: store zone-specific maps, directories, and geometry

### CellsPlotter (`cells_plotter.py`)

- Purpose: visualize segmentation and coverage paths

## 5. Segmentation Module (`covpp/segmentation/`)

Splits each zone into planning cells.

### SegmentationHandler

- Purpose: load the configured segmentation plugin and convert base cells into planning cells
- Key outputs: `bcells`, `cells`, `n_cells`

### SegmentationBase (`segmentation_base.py`)

- Purpose: base interface for segmentation plugins
- Required method: `run()`

### Segmentation plugins

| Name | Entry Point | Description |
| --- | --- | --- |
| `simple` | `covpp.segmentation.simple_segmentation.segmentation:Segmentation` | Single-cell zone |
| `voronoi` | `covpp.segmentation.voronoi_segmentation.segmentation:Segmentation` | Voronoi partitioning |
| `skeleton` | `covpp.segmentation.skeleton_segmentation.segmentation:Segmentation` | Skeleton-based decomposition |
| `distance_transform` | `covpp.segmentation.distance_transform_segmentation` | Distance transform with thresholding |
| `morphological` | `covpp.segmentation.morphological_segmentation` | Erosion-based room detection |

### CellBase (`cell_base.py`)

- Purpose: geometric cell representation

### PCell (`cell_planning.py`)

- Purpose: processed cell with path and sequencing metadata

### MapPolygons (`map_polygons.py`)

- Purpose: polygon representation of the workspace boundary and obstacles

## 6. Coverage Planning Module (`covpp/coverage_planning/`)

Generates a coverage path for each cell.

### CoveragePPHandler

- Purpose: load the configured coverage planner and run it for every cell

### CoveragePathPlannerBase (`coverage_pp_base.py`)

- Purpose: base class for cell-level coverage planners
- Required method: `coverage_planning()`
- Common helpers: path filtering, rotation, and metric conversion

### Coverage planners

| Name | Entry Point | Technique |
| --- | --- | --- |
| `boust` | `covpp.coverage_planning.boustrophedon.coverage_pp_boust:CoveragePathPlanner` | Back-and-forth sweeping |
| `contour` | `covpp.coverage_planning.contour.coverage_pp_contour:CoveragePathPlanner` | Contour-following planning |
| `grid_energy_min` | `covpp.coverage_planning.local_energy_minimization.cov_pp_local_energy_min:CoveragePathPlanner` | Grid-based local-energy minimization |

### Coverage planner notes

- `boust`: suited to rectilinear spaces and predictable sweeps.
- `contour`: suited to irregular rooms and boundary-following coverage.
- `grid_energy_min`: suited to noisy or cluttered maps where a greedy grid sweep is acceptable.

## 7. Path Planning Module (`covpp/path_planning/`)

Provides low-level path search between two map points.

### PathPlannerBase (`path_planner_base.py`)

- Purpose: build a cost map, validate endpoints, and convert between local and global coordinates
- Required method: `plan_path(start, end)`

### Path planner plugins

| Name | Entry Point | Technique |
| --- | --- | --- |
| `astar` | `covpp.path_planning.path_planner_pyastar:PathPlanner` | Grid-based A* search via `pyastar2d` |

## 8. Sequencing Module (`covpp/sequence/`)

Orders the cells and connects their paths.

### SequenceHandler

- Purpose: compute cell visitation order and build the full connected path
- Key outputs: `paths_px`, `paths_m`, `c_paths_px`, `c_paths_m`, `f_path_px`, `f_path_m`

### ConnectPaths (`connect_paths.py`)

- Purpose: generate collision-aware paths between consecutive cells

### SeqOptimizerBase (`sequence_optimizer_base.py`)

- Purpose: shared optimizer scaffolding
- Required method: `create_tour()`

### Sequence optimizers

| Name | Entry Point | Approach |
| --- | --- | --- |
| `dfs` | `covpp.sequence.dfs_optimizer.sequence_optimizer:SeqOptimizer` | Depth-first traversal |
| `milp` | `covpp.sequence.milp_optimizer.sequence_optimizer:SeqOptimizer` | OR-Tools MILP solver |

### SequencePaths (`sequence_handler.py`)

- Purpose: store coverage, connection, and full paths

## 9. Evaluation Module (`covpp/evaluation/`)

Computes zone and district metrics.

### ZoneEvaluationHandler

- Purpose: evaluate the final zone path bundle

### ZoneEvaluationBase (`zone_evaluation_base.py`)

- Purpose: base class for zone evaluators and the `ZoneStatsBase` data structure

### DistrictEvaluationHandler (`district_evaluation.py`)

- Purpose: aggregate zone metrics into district metrics

### Typical metrics

- Travel time
- Path length
- Yaw or heading change
- Cleaned area and overlap area
- Planning time per stage
