# Module Details

## 1. Base Module (`covpp/base/`)

Core utilities and data structures used throughout the pipeline.

### **InputsProcessor**

- **Purpose**: Load and validate input data
- **Key Methods**:
  - `__init__(map_name)`: Load map and configs
  - `set_paths()`: Define file paths
  - `load_yaml_file()`: Read YAML configuration
  - `update_map_configs()`: Update MapConfig object

### **MapImageProcessor**

- **Purpose**: Process raw map images
- **Key Methods**:
  - `__init__(map_arr, map_config)`: Initialize with map array and MapConfig
  - `process_map()`: Apply morphological operations
  - `set_map_data()`: Create MapData object
  - `complete_map_data()`: Add metadata

### **MapData**

- **Purpose**: Store map information
- **Attributes**:
  - `orig_map`: Original map array
  - `up_bin_map`: Processed binary map
  - `resolution`: Map resolution (m/pixel)
  - `updated_w/h`: Map dimensions

### **Configs** (`configs.py`)

Configuration classes for different pipeline stages:

- **MapConfig**: Map-specific configuration (resolution, directories, paths)
- **PipeLineConfig**: Main pipeline settings (method selection, plotting options)
- **PlanningConfig**: Path planning parameters (robot dimensions, coverage settings)
- **SpaceConfig**: Space analysis settings (morphological operations, filtering)

### **MyUtils** (my_utils.py)

Utility functions and classes:

- **MyPoint**: Point representation with x, y coordinates
- **MyPath**: Path as sequence of points
- `get_n_colors_cv()`: Generate color palette for OpenCV
- `get_n_colors_plt()`: Generate color palette for matplotlib

### **Logger** (logger.py)

Custom logging with colored output:

- `info()`, `info1()`, `info2()`: Information messages
- `warn()`: Warning messages
- `error()`: Error messages

## 2. GUI / Interaction Module (`covpp/gui/`)

Interactive tools for district selection and visualization.

### **DistrictGen** (create_district_simple.py)

- **Purpose**: Create district rectangle
- **Key Methods**:
  - `__init__(map_data)`: Initialize with map data
  - `get_district_rect()`: Generate Rectangle object for district
- **Note**: An interactive version exists in create_district.py

### **Rectangle** (rectangle.py)

- **Purpose**: Represent rectangular regions
- **Attributes**:
  - `xc, yc`: Center coordinates
  - `w, h`: Width and height
  - `rot`: Rotation angle
  - `vertices_t`: Corner vertices

## 3. Space Analysis Module (`covpp/spa/`)

Analyzes workspace and creates feasible position maps.

### **SpaceAnalyzer**

- **Purpose**: Process selected district into usable workspace
- **Key Methods**:
  - `__init__(map_data, rectangle)`: Initialize with map and district
  - `create_local_map_ws_selection()`: Generate local workspace map
  - `update_by_rectangle()`: Update for zone-specific analysis
  - `create_zone_map_data()`: Create ZoneMapData object
- **Key Attributes**:
  - `zone_map_data`: Processed zone/district data
  - `vertices`: Boundary vertices
  - `binary_map_0`: Input binary map

### **spa_utils.py**

Utility functions:

- `find_circumscribed_rectangle()`: Find bounding rectangle
- Helper functions for geometric computations

## 4. District Module (`covpp/district/`)

Manages district and zone-level operations.

### **DistrictData**

- **Purpose**: High-level district management
- **Key Methods**:
  - `__init__(spa)`: Initialize with SpaceAnalyzer
  - `create_zones_hl()`: Create all zones
  - `run_planning_pipeline()`: Execute planning for all zones
  - `update_map_planning()`: Merge zone results
  - `save_map_planning()`: Save output maps
- **Key Attributes**:
  - `dmd`: District map data (ZoneMapData)
  - `zone_datas`: List of ZoneData objects
  - `stats`: District-level statistics

### **ZoneData** (zone.py)

- **Purpose**: Manage individual zone operations
- **Key Methods**:
  - `__init__(ind, zmd)`: Initialize zone with index and map data
  - `zone_planning_pipeline()`: Run full planning pipeline for zone
  - `post_planning()`: Update maps and statistics
- **Key Attributes**:
  - `zmd`: Zone map data
  - `cells`: List of segmentation cells
  - `segmentation_hl`: Segmentation interface
  - `cov_path_planner`: Coverage planning interface
  - `sequence`: Path sequencing handler
  - `stats`: Zone statistics

### **ZoneMapData** (zone_map_data.py)

- **Purpose**: Store zone-specific map data
- **Attributes**:
  - `map_sf_orig`: Selected free space (original)
  - `map_ws`: Workspace map
  - `map_planning`: Planning visualization
  - `vertices`: Zone boundary vertices
  - `resolution`: Map resolution
  - `method_results_dir`: Output directory path

### **CellsPlotter** (cells_plotter.py)

- **Purpose**: Visualization of planning results
- **Key Methods**:
  - `plot_paths()`: Display coverage paths
  - `plot_cells()`: Show cell decomposition

## 5. Segmentation Module (`covpp/segmentation/`)

Decomposes zones into coverage cells.

### **SegmentationHandler**

- **Purpose**: High-level segmentation interface
- **Key Methods**:
  - `__init__(zmd)`: Initialize with zone map data
  - `run()`: Execute segmentation based on selected method
- **Key Attributes**:
  - `segmentor`: Actual segmentation algorithm
  - `bcells`: Base cells from segmentation
  - `cells`: Processed cells (PCell objects)
  - `map_polygons`: Polygon representation of map

### **SegmentationBase** (segmentation_base.py)

- **Purpose**: Base class for segmentation algorithms
- **Methods to override**:
  - `run()`: Implement specific segmentation logic

### **Segmentation Methods**

Segmentation algorithms are discovered via the `segmentors` entry point group:

| Name      | Entry Point                                                                 | Description                |
| --------- | --------------------------------------------------------------------------- | -------------------------- |
| `simple`  | `covpp.segmentation.simple_segmentation.segmentation_simple:Segmentation`   | Single cell (whole zone)   |
| `voronoi` | `covpp.segmentation.voronoi_segmentation.segmentation_voronoi:Segmentation` | Voronoi-based partitioning |

### **CellBase** (cell_base.py)

- **Purpose**: Basic cell representation
- **Attributes**:
  - `id`: Cell identifier
  - `polygon`: Cell boundary polygon
  - `centroid`: Cell center point

### **PCell** (cell_processed.py)

- **Purpose**: Processed cell with planning information
- **Extends**: CellBase
- **Additional Attributes**:
  - `path_m`: Coverage path in meters
  - `path_px`: Coverage path in pixels
  - `adjacent_ids`: Adjacent cell IDs
  - `do_planning`: Flag for planning requirement

### **MapPolygons** (map_polygons.py)

- **Purpose**: Polygon representation of workspace
- **Attributes**:
  - `b_poly`: Boundary polygon
  - `obs_polys`: Obstacle polygons

## 6. Coverage Planning Module (`covpp/planning/`)

Generates coverage paths for individual cells.

### **CoveragePPHandler**

- **Purpose**: Manage coverage planning for all cells
- **Key Methods**:
  - `__init__(zone_map_data)`: Initialize with zone data
  - `plan(cells)`: Run planning for all cells
  - `all_cells_covpp()`: Apply coverage planner to each cell
- **Key Attributes**:
  - `cells`: Input cells from segmentation
  - `all_cell_covp_planners`: List of cell planners

### **CoveragePathPlannerBase** (coverage_pp_base.py)

- **Purpose**: Base class for coverage planners
- **Key Methods**:
  - `__init__(cell, zmd)`: Initialize with cell and zone data
  - `coverage_planning()`: Main planning method (to be overridden)
  - `point_px_to_m()`: Convert pixel to meter coordinates
  - `rotate_path()`: Apply rotation transformation
- **Key Attributes**:
  - `cell`: Target cell
  - `path_m`: Result path in meters
  - `path_px`: Result path in pixels

Coverage planners are discovered via the `planners` entry point group:

| Name              | Entry Point                                                                                     | Technique                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `boust`           | `covpp.coverage_planning.boustrophedon.coverage_pp_boust:CoveragePathPlanner`                   | Boustrophedon (sweeping / trapezoids + back-and-forth lines) |
| `contour`         | `covpp.coverage_planning.contour.coverage_pp_contour:CoveragePathPlanner`                       | Inward contour / distance transform / rectangle hybrid       |
| `grid_energy_min` | `covpp.coverage_planning.local_energy_minimization.cov_pp_local_energy_min:CoveragePathPlanner` | Grid-based coverage using local energy minimization          |

Key boustrophedon submodules: `decomposition_bs_mh.py`, `cell_cov_planner.py`, `obstacles.py`, `events.py`, `plotter.py`.

## 7. Path Planning Module (`covpp/path_planning/`)

Provides low-level graph search utilities used whenever two points inside a zone
need to be connected (e.g., ConnectPaths, evaluation overlays).

### **PathPlannerBase** (path_planner_base.py)

- **Purpose**: Shared logic for constructing cost maps, validating endpoints, and converting between global/local indices.
- **Process**:
  1. Build a float cost map from the workspace slice (free vs. occupied values).
  2. Convert global pixel coordinates into the planner’s local frame.
  3. Delegate to the concrete planner implementation via `plan_path()`.
  4. Lift the resulting local polyline back to global coordinates and wrap it in `MyPath`.
- **Result**: `find_path()` returns a `MyPath` whose points live in the zone’s global pixel frame, ready for connection or distance measurements.

### **Path Planner Plugins** (entry-point group: `path_planners`)

| Name    | Entry Point                                            | Technique / Use Case                                                                 |
| ------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `astar` | `covpp.path_planning.path_planner_pyastar:PathPlanner` | Grid-based A\* search via `pyastar2d`, used by `ConnectPaths` to link cell endpoints |

Adding new planners (e.g., D\* Lite, Jump Point Search) only requires inheriting
`PathPlannerBase`, implementing `plan_path()`, and registering the class under
the `path_planners` entry-point group in `pyproject.toml`.

---

## 8. Sequencing Module (`covpp/sequence/`)

Optimizes cell visit order and connects paths.

### **SequenceHandler**

- **Purpose**: Create optimal cell traversal sequence
- **Key Methods**:
  - `__init__(zmd)`: Initialize with zone map data
  - `run(cells, map_polygons)`: Execute sequencing
  - `optimize_sequence()`: Find optimal cell order
  - `connect_paths()`: Create connections between cells
  - `create_full_path()`: Generate complete path
- **Key Attributes**:
  - `paths_m`: Coverage paths in meters
  - `c_paths_m`: Connection paths in meters
  - `f_path_m`: Full connected path
  - `seq_idxs`: Optimized cell sequence

### **ConnectPaths** (connect_paths.py)

- **Purpose**: Generate paths connecting cells
- **Key Methods**:
  - `connect(start, end)`: Create connection path
  - Uses pathfinding algorithms (A\*, Dijkstra)

### **SeqOptimizerBase** (`sequence_optimizer_base.py`)

- **Purpose**: Provide shared bookkeeping (cell indices, optional distance matrices, start cell) and utility helpers for all sequencing strategies.
- **Result**: Subclasses implement `create_tour()` to populate `tour_ids` / `tour_idxs`, after which `SequenceHandler` writes `opt_id` / `next_cell_id` back into each `PCell`.

### **Sequence Optimizers** (`sequencers` entry point group)

| Name   | Entry Point                                                     | Approach                                                 |
| ------ | --------------------------------------------------------------- | -------------------------------------------------------- |
| `dfs`  | `covpp.sequence.dfs_optimizer.sequence_optimizer:SeqOptimizer`  | Depth-first traversal of the cell adjacency graph        |
| `milp` | `covpp.sequence.milp_optimizer.sequence_optimizer:SeqOptimizer` | Mixed Integer Linear Programming TSP solver via OR-Tools |

> A simple pass-through optimizer exists in `covpp.sequence.simple_optimizer`, but it is not registered under `sequencers` by default. Add it to `pyproject.toml` if you need the original ordering as a selectable plugin.

### **SequencePaths** (sequence.py)

- **Purpose**: Container for path collections
- **Attributes**:
  - `paths_px`: Cell paths in pixels
  - `paths_m`: Cell paths in meters
  - `c_paths_px`: Connection paths in pixels
  - `c_paths_m`: Connection paths in meters
  - `f_path_px`: Full path in pixels
  - `f_path_m`: Full path in meters

## 9. Evaluation Module (`covpp/evaluation/`)

Computes performance metrics and statistics.

### **ZoneEvaluationHandler**

- **Purpose**: Evaluate zone-level performance
- **Key Methods**:
  - `__init__(zmd)`: Initialize with zone data
  - `evaluate(paths, duration)`: Compute metrics
- **Metrics**:
  - Path length
  - Coverage percentage
  - Number of yaw changes
  - Planning time

### **DistrictEvaluationHandler**

- **Purpose**: Aggregate zone statistics to district level
- **Key Methods**:
  - `evaluate(zone_stats)`: Aggregate statistics
  - Save results to JSON

### **ZoneStatsBase** (zone_evaluation_base.py)

- **Purpose**: Data structure for zone statistics
- **Attributes**:
  - `path_len_m`: Total path length
  - `coverage_p`: Coverage percentage
  - `n_paths`: Number of paths
  - `yaw_changes`: Number of direction changes

### **DistrictStatsBase**

- **Purpose**: Data structure for district statistics
- Similar to ZoneStatsBase but aggregated
