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

### **Geometry** (geometry.py)

- **MyPoint**: Point representation with x, y coordinates
- **MyPath**: Path as sequence of points
- Other helpers (e.g., polygon edge utilities)

### **BaseUtils** (base_utils.py)

- `get_n_colors_cv()`: Generate color palette for OpenCV
- `get_n_colors_plt()`: Generate color palette for matplotlib

### **Logger** (logger.py)

Custom logging with colored output:

- `info()`, `info1()`, `info2()`: Information messages
- `warn()`: Warning messages
- `error()`: Error messages

## 2. GUI / Interaction Module (`covpp/gui/`)

Interactive primitives used by district selection and visualization.

### **Rectangle** (rectangle.py)

- **Purpose**: Represent rectangular regions
- **Attributes**:
  - `xc, yc`: Center coordinates
  - `w, h`: Width and height
  - `rot`: Rotation angle
  - `vertices_t`: Corner vertices

### **RecEventHandler** (rect_event_hl.py)

- **Purpose**: Mouse/keyboard event handler for interactive rectangle selection
- **Notes**: Used by `DistrictGen` to let users draw/adjust the district.

### **VisualRectangle** (visual_rectangle.py)

- **Purpose**: Rendering helpers for rectangles in OpenCV canvases
- **Notes**: Provides visual feedback during GUI selection.

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

### **spa_rectangles.py**

Utility functions:

- `find_circumscribed_rectangle()`: Find global bounding (min-area) rectangle
- `find_inscribed_rotated_rectangle()`: Heuristic for maximum-area inscribed rectangle

## 4. District Module (`covpp/district/`)

Manages district and zone-level operations.

### **DistrictGen** (create_district.py)

- **Purpose**: Create a district rectangle via GUI or whole-map fallback
- **Key Methods**:
  - `__init__(map_data)`: Initialize and choose GUI vs. full-map
  - `create_district_gui()`: Interactive selection using `RecEventHandler`
  - `create_district_whole()`: Non-interactive full-map rectangle
  - `get_district_rect()`: Return the created `Rectangle`

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

### **PCell** (cell_planning.py)

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

## 6. Coverage Planning Module (`covpp/coverage_planning/`)

Generates coverage paths for individual cells.

### **CoveragePPHandler**

- **Purpose**: Manage coverage planning for all cells
- **Key Methods**:
  - `__init__(zone_map_data)`: Initialize with zone data
  - `run(cells)`: Run planning for all cells
  - `run_covpp_method()`: Apply configured planner to each cell
- **Key Attributes**:
  - `cells`: Input cells from segmentation
  - (Planners are loaded per cell; results stored back into each `PCell`)

### **CoveragePathPlannerBase** (coverage_pp_base.py)

- **Purpose**: Base class for coverage planners
- **Key Methods**:
  - `__init__(cell, zmd)`: Initialize with cell and zone data
  - `run()`: Execute full planning pipeline (calls `coverage_planning()`)
  - `coverage_planning()`: Main planning method (to be overridden)
  - `apply_distance_filter_path()`: Enforce waypoint spacing
  - `rotate_path()`: Apply rotation transformation
- **Key Attributes**:
  - `cell`: Target cell
  - `path_m`: Result path in meters
  - `path_px`: Result path in pixels

Coverage planners are discovered via the `coverage_planners` entry point group:

| Name              | Entry Point                                                                                     | Technique                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `boust`           | `covpp.coverage_planning.boustrophedon.coverage_pp_boust:CoveragePathPlanner`                   | Boustrophedon (sweeping / trapezoids + back-and-forth lines) |
| `contour`         | `covpp.coverage_planning.contour.coverage_pp_contour:CoveragePathPlanner`                       | Inward contour / distance transform / rectangle hybrid       |
| `grid_energy_min` | `covpp.coverage_planning.local_energy_minimization.cov_pp_local_energy_min:CoveragePathPlanner` | Grid-based coverage using local energy minimization          |

Key boustrophedon submodules: `decomposition_bs_mh.py`, `cell_cov_planner.py`, `obstacles.py`, `events.py`, `plotter.py`.

One-liners (when to use):

- `boust`: Reliable for large, rectilinear spaces with few obstacles; produces tidy, parallel lanes with predictable turn behavior.
- `contour`: Best for organic/curvy rooms; iteratively follows inward contours (buffer + distance transform) to hug boundaries.
- `grid_energy_min`: Simple greedy sweep on a grid that minimizes heading changes; topology-agnostic and robust to noisy maps.

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
  - `run(cells, map_polygons)`: Execute sequencing pipeline
  - `create_sequence()`: Prepare connections and collect paths in order
  - `optimize_sequence(distances=None)`: Find optimal cell order via plugin
  - `connect_paths()`: Create connections between consecutive cells
  - `create_full_path()`: Generate complete path bundle
- **Key Attributes**:
  - `paths_m`: Coverage paths in meters
  - `c_paths_m`: Connection paths in meters
  - `f_path_m`: Full connected path
  - `seq_idxs`: Optimized cell sequence

### **ConnectPaths** (connect_paths.py)

- **Purpose**: Generate collision-free paths between cell endpoints
- **Key Methods**:
  - `initialize(cells)`: Allocate connection and distance matrices
  - `find_all_connections(cells)`: Precompute pairwise paths
  - `find_all_distances()`: Compute pairwise path lengths
  - `connect_paths(paths_px, seq_idxs=None)`: Build sequence connections
  - `find_path(start, end)`: Find a single inter-cell path
  - `update_connection_2_cells(i, j, start, end)`: Refresh one entry

### **SeqOptimizerBase** (`sequence_optimizer_base.py`)

- **Purpose**: Provide shared bookkeeping (cell indices, optional distance matrices, start cell) and utility helpers for all sequencing strategies.
- **Result**: Subclasses implement `create_tour()` to populate `tour_ids` / `tour_idxs`, after which `SequenceHandler` writes `opt_id` / `next_cell_id` back into each `PCell`.

### **Sequence Optimizers** (`sequencers` entry point group)

| Name   | Entry Point                                                     | Approach                                                 |
| ------ | --------------------------------------------------------------- | -------------------------------------------------------- |
| `dfs`  | `covpp.sequence.dfs_optimizer.sequence_optimizer:SeqOptimizer`  | Depth-first traversal of the cell adjacency graph        |
| `milp` | `covpp.sequence.milp_optimizer.sequence_optimizer:SeqOptimizer` | Mixed Integer Linear Programming TSP solver via OR-Tools |

> A simple pass-through optimizer exists in `covpp.sequence.simple_optimizer`, but it is not registered under `sequencers` by default. Add it to `pyproject.toml` if you need the original ordering as a selectable plugin.

### **SequencePaths** (sequence_handler.py)

- **Purpose**: Container for path collections
- **Attributes**:
  - `paths_px`: Cell paths in pixels
  - `paths_m`: Cell paths in meters
  - `c_paths_px`: Connection paths in pixels
  - `c_paths_m`: Connection paths in meters
  - `f_path_px`: Full path in pixels
  - `f_path_m`: Full path in meters
  - `n_paths`: Number of coverage paths

## 9. Evaluation Module (`covpp/evaluation/`)

Computes performance metrics and statistics.

### **ZoneEvaluationHandler**

- **Purpose**: Evaluate zone-level performance
- **Key Methods**:
  - `__init__(zmd)`: Initialize with zone data
  - `evaluate(paths, times)`: Compute metrics and persist results
- **Metrics (ZoneStatsBase)**:
  - `map_name`, `method`, `zone_ind`
  - Travel times: `trav_time_full`, `trav_time_coverage`
  - Stage times: `ptime_segmentation`, `ptime_coverage_planning`, `ptime_sequencing`, `ptime_pipeline`
  - Path metrics: `full_length`, `coverage_length`, `yaw_path_full`, `yaw_path_coverage`
  - Areas: `cleaning_area_px/m`, `cleaned_area_px/m`, `area_uncleaned_m`, `area_overlaped_m`
  - Ratios: `area_coverage_pc`, `area_overlap_pc`

### **DistrictEvaluationHandler**

- **Purpose**: Aggregate zone statistics to district level
- **Key Methods**:
  - `evaluate(zone_stats)`: Aggregate statistics
  - Save results to JSON

### **ZoneStatsBase** (zone_evaluation_base.py)

- **Purpose**: Data structure for zone KPIs and serialization
- **See**: Metrics listed under ZoneEvaluationHandler

### **DistrictStatsBase** (district_evaluation.py)

- **Purpose**: Aggregated district-level KPIs and serialization
- **Key Fields**: Mirrors sums/ratios across zones (`l_path_full`, `l_path_coverage`, times, areas, coverage/overlap ratios)
