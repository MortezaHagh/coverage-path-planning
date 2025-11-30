# Coverage Path Planning Pipeline Documentation

Last Updated: 2025-11-09

---

## Quick Summary

`covpp` is a modular, plugin-driven coverage path planning pipeline for indoor (grid / occupancy) maps. Starting from a grayscale map + YAML, it selects a district, analyzes free space, segments into cells, plans per–cell coverage paths, sequences them, and evaluates performance. Algorithms for segmentation, coverage planning, low-level path finding, and sequencing are loaded via Python entry points (`segmentors`, `planners`, `path_planners`, `sequencers`) so you can add new methods without touching core code.

---

## Overview

This is a comprehensive coverage path planning system that generates optimal paths for robotic coverage of indoor environments. The pipeline processes map images, performs space segmentation, plans coverage paths, and optimizes the sequence of path execution.

---

## Table of content

- [Quick Summary](#quick-summary)
- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Main Workflow (`covpp/main.py`)](#main-workflow-covppmainpy)
- [Module Details](#module-details)
- [Data Flow](#data-flow)
- [Configuration & Parameters](#configuration--parameters)
- [Output Artifacts](#output-artifacts)
- [Plugin-Based Architecture (Entry Points)](#plugin-based-architecture-entry-points)
- [Adding New Methods (Plugin Development)](#adding-new-methods-plugin-development)
- [Benefits of the Plugin Architecture](#benefits-of-the-plugin-architecture)
- [End-to-End Pattern](#end-to-end-pattern)
- [Dependencies](#dependencies)
- [Usage Examples](#usage-examples)
- [References](#references)

## Pipeline Architecture

The system follows a **modular, plugin-based architecture** where new algorithms can be easily integrated by extending base classes. Each major component (Segmentation, Coverage Planning, Sequence Optimization, and Evaluation) is designed with abstract base classes that define the interface, allowing developers to add new methods without modifying the core pipeline.

```text
PGM + YAML → Inputs → Image Processing → District & Space Analysis → Zone(s)
  → Segmentation (cells) → Coverage Planning (paths)
    → Sequencing (order + connections) → Evaluation → Results (figures + JSON)
```

**Key Architectural Principles:**

- **Extensibility**: Add new methods by inheriting from base classes
- **Pluggability**: Switch between algorithms via configuration files
- **Separation of Concerns**: Each module has a clear, single responsibility
- **Handler Pattern**: High-level handlers dynamically load specific implementations
- **Configuration-Driven**: Method selection and parameters controlled through config files

---

## Main Workflow (`covpp/main.py`)

The pipeline executes in the following sequential steps:

### 1. **Input Processing** (`InputsProcessor`)

- Reads map image (PGM format) and configuration files (YAML/JSON)
- Validates input data and directory structure
- Initializes base configurations

### 2. **Map Image Processing** (`MapImageProcessor`)

- Converts raw map to binary format (free/occupied space)
- Applies morphological operations (erosion/dilation)
- Generates workspace map considering robot dimensions

### 3. **District Generation** (`DistrictGen` – simple version)

- Defines the coverage area (district) within the map
- Creates bounding rectangle for the coverage region
- Supports whole-map or partial-map selection

### 4. **Space Analysis** (`SpaceAnalyzer`)

- Analyzes the selected district for feasible robot positions
- Creates local workspace maps
- Processes free space and computes accessibility

### 5. **District Data Creation** (`DistrictData`)

- Initializes district-level data structures
- Creates zones (subdivisions of the district)
- Manages zone map data

### 6. **Zone-Level Coverage Planning** (`ZoneData`)

For each zone in the district:

- **Segmentation**: Divides zone into cells
- **Coverage Planning**: Generates paths for each cell
- **Sequencing**: Optimizes cell visit order and connects paths
- **Evaluation**: Computes statistics and metrics

---

## Module Details

### 1. Base Module (`covpp/base/`)

Core utilities and data structures used throughout the pipeline.

#### **InputsProcessor**

- **Purpose**: Load and validate input data
- **Key Methods**:
  - `__init__(map_name)`: Load map and configs
  - `set_paths()`: Define file paths
  - `load_yaml_file()`: Read YAML configuration
  - `update_map_configs()`: Update MapConfig object

#### **MapImageProcessor**

- **Purpose**: Process raw map images
- **Key Methods**:
  - `__init__(map_arr, map_config)`: Initialize with map array and MapConfig
  - `process_map()`: Apply morphological operations
  - `set_map_data()`: Create MapData object
  - `complete_map_data()`: Add metadata

#### **MapData**

- **Purpose**: Store map information
- **Attributes**:
  - `orig_map`: Original map array
  - `up_bin_map`: Processed binary map
  - `resolution`: Map resolution (m/pixel)
  - `updated_w/h`: Map dimensions

#### **Configs** (`configs.py`)

Configuration classes for different pipeline stages:

- **MapConfig**: Map-specific configuration (resolution, directories, paths)
- **PipeLineConfig**: Main pipeline settings (method selection, plotting options)
- **PlanningConfig**: Path planning parameters (robot dimensions, coverage settings)
- **SpaceConfig**: Space analysis settings (morphological operations, filtering)

#### **MyUtils** (my_utils.py)

Utility functions and classes:

- **MyPoint**: Point representation with x, y coordinates
- **MyPath**: Path as sequence of points
- `get_n_colors_cv()`: Generate color palette for OpenCV
- `get_n_colors_plt()`: Generate color palette for matplotlib

#### **Logger** (logger.py)

Custom logging with colored output:

- `info()`, `info1()`, `info2()`: Information messages
- `warn()`: Warning messages
- `error()`: Error messages

---

### 2. GUI / Interaction Module (`covpp/gui/`)

Interactive tools for district selection and visualization.

#### **DistrictGen** (create_district_simple.py)

- **Purpose**: Create district rectangle
- **Key Methods**:
  - `__init__(map_data)`: Initialize with map data
  - `get_district_rect()`: Generate Rectangle object for district
- **Note**: An interactive version exists in create_district.py

#### **Rectangle** (rectangle.py)

- **Purpose**: Represent rectangular regions
- **Attributes**:
  - `xc, yc`: Center coordinates
  - `w, h`: Width and height
  - `rot`: Rotation angle
  - `vertices_t`: Corner vertices

---

### 3. Space Analysis Module (`covpp/spa/`)

Analyzes workspace and creates feasible position maps.

#### **SpaceAnalyzer**

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

#### **spa_utils.py**

Utility functions:

- `find_circumscribed_rectangle()`: Find bounding rectangle
- Helper functions for geometric computations

---

### 4. District Module (`covpp/district/`)

Manages district and zone-level operations.

#### **DistrictData**

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

#### **ZoneData** (zone.py)

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

#### **ZoneMapData** (zone_map_data.py)

- **Purpose**: Store zone-specific map data
- **Attributes**:
  - `map_sf_orig`: Selected free space (original)
  - `map_ws`: Workspace map
  - `map_planning`: Planning visualization
  - `vertices`: Zone boundary vertices
  - `resolution`: Map resolution
  - `method_results_dir`: Output directory path

#### **CellsPlotter** (cells_plotter.py)

- **Purpose**: Visualization of planning results
- **Key Methods**:
  - `plot_paths()`: Display coverage paths
  - `plot_cells()`: Show cell decomposition

---

### 5. Segmentation Module (`covpp/segmentation/`)

Decomposes zones into coverage cells.

#### **SegmentationHandler**

- **Purpose**: High-level segmentation interface
- **Key Methods**:
  - `__init__(zmd)`: Initialize with zone map data
  - `run()`: Execute segmentation based on selected method
- **Key Attributes**:
  - `segmentor`: Actual segmentation algorithm
  - `bcells`: Base cells from segmentation
  - `cells`: Processed cells (PCell objects)
  - `map_polygons`: Polygon representation of map

#### **SegmentationBase** (segmentation_base.py)

- **Purpose**: Base class for segmentation algorithms
- **Methods to override**:
  - `run()`: Implement specific segmentation logic

#### **Segmentation Methods**

Segmentation algorithms are discovered via the `segmentors` entry point group:

| Name      | Entry Point                                                                 | Description                |
| --------- | --------------------------------------------------------------------------- | -------------------------- |
| `simple`  | `covpp.segmentation.simple_segmentation.segmentation_simple:Segmentation`   | Single cell (whole zone)   |
| `voronoi` | `covpp.segmentation.voronoi_segmentation.segmentation_voronoi:Segmentation` | Voronoi-based partitioning |

#### **CellBase** (cell_base.py)

- **Purpose**: Basic cell representation
- **Attributes**:
  - `id`: Cell identifier
  - `polygon`: Cell boundary polygon
  - `centroid`: Cell center point

#### **PCell** (cell_processed.py)

- **Purpose**: Processed cell with planning information
- **Extends**: CellBase
- **Additional Attributes**:
  - `path_m`: Coverage path in meters
  - `path_px`: Coverage path in pixels
  - `adjacent_ids`: Adjacent cell IDs
  - `do_planning`: Flag for planning requirement

#### **MapPolygons** (map_polygons.py)

- **Purpose**: Polygon representation of workspace
- **Attributes**:
  - `b_poly`: Boundary polygon
  - `obs_polys`: Obstacle polygons

---

### 6. Coverage Planning Module (`covpp/planning/`)

Generates coverage paths for individual cells.

#### **CoveragePPHandler**

- **Purpose**: Manage coverage planning for all cells
- **Key Methods**:
  - `__init__(zone_map_data)`: Initialize with zone data
  - `plan(cells)`: Run planning for all cells
  - `all_cells_covpp()`: Apply coverage planner to each cell
- **Key Attributes**:
  - `cells`: Input cells from segmentation
  - `all_cell_covp_planners`: List of cell planners

#### **CoveragePathPlannerBase** (coverage_pp_base.py)

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

---

### 7. Path Planning Module (`covpp/path_planning/`)

Provides low-level graph search utilities used whenever two points inside a zone
need to be connected (e.g., ConnectPaths, evaluation overlays).

#### **PathPlannerBase** (path_planner_base.py)

- **Purpose**: Shared logic for constructing cost maps, validating endpoints, and converting between global/local indices.
- **Process**:
  1. Build a float cost map from the workspace slice (free vs. occupied values).
  2. Convert global pixel coordinates into the planner’s local frame.
  3. Delegate to the concrete planner implementation via `plan_path()`.
  4. Lift the resulting local polyline back to global coordinates and wrap it in `MyPath`.
- **Result**: `find_path()` returns a `MyPath` whose points live in the zone’s global pixel frame, ready for connection or distance measurements.

#### **Path Planner Plugins** (entry-point group: `path_planners`)

| Name    | Entry Point                                            | Technique / Use Case                                                                 |
| ------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `astar` | `covpp.path_planning.path_planner_pyastar:PathPlanner` | Grid-based A\* search via `pyastar2d`, used by `ConnectPaths` to link cell endpoints |

Adding new planners (e.g., D\* Lite, Jump Point Search) only requires inheriting
`PathPlannerBase`, implementing `plan_path()`, and registering the class under
the `path_planners` entry-point group in `pyproject.toml`.

---

### 8. Sequencing Module (`covpp/sequence/`)

Optimizes cell visit order and connects paths.

#### **SequenceHandler**

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

#### **ConnectPaths** (connect_paths.py)

- **Purpose**: Generate paths connecting cells
- **Key Methods**:
  - `connect(start, end)`: Create connection path
  - Uses pathfinding algorithms (A\*, Dijkstra)

#### **SeqOptimizerBase** (`sequence_optimizer_base.py`)

- **Purpose**: Provide shared bookkeeping (cell indices, optional distance matrices, start cell) and utility helpers for all sequencing strategies.
- **Result**: Subclasses implement `create_tour()` to populate `tour_ids` / `tour_idxs`, after which `SequenceHandler` writes `opt_id` / `next_cell_id` back into each `PCell`.

#### **Sequence Optimizers** (`sequencers` entry point group)

| Name   | Entry Point                                                     | Approach                                                 |
| ------ | --------------------------------------------------------------- | -------------------------------------------------------- |
| `dfs`  | `covpp.sequence.dfs_optimizer.sequence_optimizer:SeqOptimizer`  | Depth-first traversal of the cell adjacency graph        |
| `milp` | `covpp.sequence.milp_optimizer.sequence_optimizer:SeqOptimizer` | Mixed Integer Linear Programming TSP solver via OR-Tools |

> A simple pass-through optimizer exists in `covpp.sequence.simple_optimizer`, but it is not registered under `sequencers` by default. Add it to `pyproject.toml` if you need the original ordering as a selectable plugin.

#### **SequencePaths** (sequence.py)

- **Purpose**: Container for path collections
- **Attributes**:
  - `paths_px`: Cell paths in pixels
  - `paths_m`: Cell paths in meters
  - `c_paths_px`: Connection paths in pixels
  - `c_paths_m`: Connection paths in meters
  - `f_path_px`: Full path in pixels
  - `f_path_m`: Full path in meters

---

### 9. Evaluation Module (`covpp/evaluation/`)

Computes performance metrics and statistics.

#### **ZoneEvaluationHandler**

- **Purpose**: Evaluate zone-level performance
- **Key Methods**:
  - `__init__(zmd)`: Initialize with zone data
  - `evaluate(paths, duration)`: Compute metrics
- **Metrics**:
  - Path length
  - Coverage percentage
  - Number of yaw changes
  - Planning time

#### **DistrictEvaluationHandler**

- **Purpose**: Aggregate zone statistics to district level
- **Key Methods**:
  - `evaluate(zone_stats)`: Aggregate statistics
  - Save results to JSON

#### **ZoneStatsBase** (zone_evaluation_base.py)

- **Purpose**: Data structure for zone statistics
- **Attributes**:
  - `path_len_m`: Total path length
  - `coverage_p`: Coverage percentage
  - `n_paths`: Number of paths
  - `yaw_changes`: Number of direction changes

#### **DistrictStatsBase**

- **Purpose**: Data structure for district statistics
- Similar to ZoneStatsBase but aggregated

---

## Data Flow

```text
main.py
  ├─> InputsProcessor
  │     └─> Load map & configs
  │
  ├─> MapImageProcessor
  │     └─> Create MapData
  │
  ├─> DistrictGen
  │     └─> Create Rectangle
  │
  ├─> SpaceAnalyzer
  │     └─> Create ZoneMapData
  │
  └─> DistrictData
        └─> create_zones_hl()
        │     └─> Create ZoneData for each zone
        │
        └─> run_planning_pipeline()
              └─> For each ZoneData:
                    ├─> SegmentationHandler.run()
                    │     └─> Create cells (PCell)
                    │
                    ├─> CoveragePPHandler.plan(cells)
                    │     └─> For each cell:
                    │           └─> CoveragePathPlanner
                    │                 └─> Generate path_m, path_px
                    │
                    ├─> SequenceHandler.run(cells, map_polygons)
                    │     ├─> Optimize cell order
                    │     ├─> Connect paths
                    │     └─> Create full path
                    │
                    └─> ZoneEvaluationHandler.evaluate()
                          └─> Compute statistics
```

---

## Configuration & Parameters

### Method Selection (`PipeLineConfig`)

- **Segmentation Methods**:

  - `simple`: Single cell (no decomposition)
  - `voronoi`: Voronoi-based decomposition

- **Coverage Planning Methods**:

  - `boust`: Boustrophedon (back-and-forth)
  - `contour`: Contour-following
  - `grid_energy_min`: Grid-based local-energy minimization

- **Path Planner Methods**:

  - `astar`: Grid A\* (pyastar2d)
  - Custom planners registered under `path_planners`

- **Sequence Optimization Methods**:
  - `dfs`: Depth-first search (`covpp.sequence.dfs_optimizer`)
  - `milp`: Mixed-integer linear programming (`covpp.sequence.milp_optimizer`)
  - `simple`: Legacy order-preserving optimizer (register `covpp.sequence.simple_optimizer` under `sequencers` to enable)

### Robot Parameters (`RobotConfig` & parts of `SpaceConfig` / `PlanningConfig`)

- `robot_radius`: Robot radius in meters
- `tool_radius`: Coverage tool radius
- `resolution`: Map resolution (m/pixel)

### Planning Parameters (`PlanningConfig`)

- `min_wp_dist`: Minimum waypoint distance
- `path_simplification`: Path smoothing options

---

## Output Artifacts

### Directory Structure

```text
results/
  └─ <map_name>/
       └─ District/
            ├─ map_planning_<method>.png
            ├─ result.json
            └─ Zone_<i>/
                 ├─ map_planning.png
                 ├─ map_segmentation.png
                 ├─ map_cleaned.png
                 ├─ result.json
                 └─ <method>/
                      ├─ paths visualization
                      └─ detailed results
```

### result.json Structure

```json
{
  "map": "Map2",
  "method": "contour",
  "zone_stats": {
    "path_len_m": 123.45,
    "coverage_p": 98.7,
    "cleaning_area_m": 50.2,
    "n_paths": 5
  },
  "planner_stats": {
    "len_m": 123.45,
    "n_paths": 5,
    "yaw_changes": 12,
    "planning_time": 2.34
  }
}
```

---

## Plugin-Based Architecture (Entry Points)

The system is built on a **plugin-based architecture** that makes it highly extensible and maintainable. Each major component in the pipeline is designed around **abstract base classes** that define the interface contract, allowing developers to implement new algorithms without modifying the existing codebase.

### Core Design Principles

#### 1. **Abstract Base Classes**

Each critical component has a base class that defines the interface:

- **`SegmentationBase`**: Interface for space decomposition algorithms
- **`CoveragePathPlannerBase`**: Interface for coverage generation inside a cell
- **`PathPlannerBase`**: Interface for low-level graph search between two map points
- **`EvaluationBase`**: Interface for performance evaluation methods
- **`SequenceOptimizer`**: Interface for path sequencing algorithms

These base classes enforce a consistent API, making it easy to swap implementations.

#### 2. **Handler Pattern + Dynamic Loading**

High-level handler classes dynamically load specific implementations based on configuration:

- **`SegmentationHandler`**:

  - Reads `segmentation_method` from config
  - Dynamically imports appropriate `Segmentation` class (simple/voronoi/custom)
  - Calls base class methods (`run()`) without knowing implementation details

- **`CoveragePPHandler`**:

  - Reads `covpp_method` from config
  - Dynamically imports appropriate `CoveragePathPlanner` class (boust/contour/custom)
  - Applies planning to all cells using consistent interface

- **`FindPathInMap` / `ConnectPaths`**:

  - Read `pp_method` from config
  - Dynamically load the appropriate low-level path planner (e.g., A\*)
  - Provide reusable point-to-point connections for coverage validation and sequencing

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
    segmentation_method: str = "voronoi"  # or "simple", "custom"
    covpp_method: str = "boust"           # or "contour", "custom"
  pp_method: str = "astar"              # or any custom path-planner plugin name
    sequence_method: str = "simple"       # or "dfs", "milp", "custom"
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

---

## Adding New Methods (Plugin Development)

The plugin-based design makes extending the system straightforward:

### 1. **Adding a New Segmentation Method** (Plugin)

**Step 1**: Create your implementation directory:

```text
scripts/segmentation/my_method/
    __init__.py
    segmentation_my_method.py
```

**Step 2**: Inherit from `SegmentationBase`:

```python
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.cell_base import CellBase

class Segmentation(SegmentationBase):
    def __init__(self, zmd, map_polygons):
        super().__init__(zmd, map_polygons)
        # Your initialization

    def run(self):
        """Implement your segmentation algorithm."""
        # Your segmentation logic
        # Create cells as list of CellBase objects
        self.cells = [CellBase(...), ...]
        self.n_cells = len(self.cells)
```

**Step 3**: Register in `pyproject.toml` entry points:

```toml
[project.entry-points.segmentors]
my_method = "covpp.segmentation.my_method.segmentation_my_method:Segmentation"
```

**Step 4**: Set configuration:

```python
# In configs file or YAML
segmentation_method: "my_method"
```

**That's it!** Your method is now integrated into the pipeline.

### 2. **Adding a New Coverage Planning Method** (Plugin)

**Step 1**: Create implementation directory:

```text
scripts/planning/my_planner/
    __init__.py
    coverage_pp_my_planner.py
```

**Step 2**: Inherit from `CoveragePathPlannerBase`:

```python
from covpp.coverage_planning.coverage_pp_base import CoveragePathPlannerBase

class CoveragePathPlanner(CoveragePathPlannerBase):
    def __init__(self, cell, zmd):
        super().__init__(cell, zmd)
        # Your initialization

    def coverage_planning(self):
        """Implement your path planning algorithm."""
        # Generate path for the cell
        self.path_px = [...]  # List of (x, y) in pixels
        self.create_path_m()  # Convert to meters (inherited method)
```

**Step 3**: Register via entry point group `planners`:

```toml
[project.entry-points.planners]
my_planner = "covpp.coverage_planning.my_planner.coverage_pp_my_planner:CoveragePathPlanner"
```

**Step 4**: Set configuration:

```python
covpp_method: "my_planner"
```

### 3. **Adding a New Path Planner** (Plugin)

**Step 1**: Create implementation directory:

```text
scripts/path_planning/my_planner/
  __init__.py
  path_planner_my_planner.py
```

**Step 2**: Inherit from `PathPlannerBase`:

```python
from covpp.path_planning.path_planner_base import PathPlannerBase

class PathPlanner(PathPlannerBase):
  def plan_path(self, start, end):
    """Implement your grid/graph search here and return local pixels."""
    # return [(x0, y0), (x1, y1), ...]
```

**Step 3**: Register via entry point group `path_planners`:

```toml
[project.entry-points.path_planners]
my_planner = "covpp.path_planning.my_planner.path_planner_my_planner:PathPlanner"
```

**Step 4**: Set configuration:

```python
pp_method: "my_planner"
```

### 4. **Adding a New Sequence Optimization Method** (Plugin)

**Step 1**: Create optimizer class:

```python
# In scripts/sequence/optimize_my_method.py
class OptimizeCellsSeq:
    def __init__(self, cells, ...):
        self.cells = cells
        # Your initialization

    def optimize(self):
        """Implement your sequencing algorithm."""
        # Return optimized sequence
        return optimized_sequence
```

**Step 2**: Register via entry point group `sequencers`:

```toml
[project.entry-points.sequencers]
my_method = "covpp.sequence.optimize_my_method:MySeqOpt"
```

**Step 3**: Set configuration:

```python
sequence_method: "my_method"
```

### 5. **Adding a New Evaluation Method**

**Step 1**: Inherit from evaluation base:

```python
# In scripts/evaluation/zone_evaluation_my_method.py
from covpp.evaluation.zone_evaluation_base import ZoneEvaluationBase

class ZoneEvaluation(ZoneEvaluationBase):
    def evaluate(self, paths, duration):
        """Implement your evaluation metrics."""
        # Compute statistics
        self.stats.custom_metric = ...
        return self.stats
```

**Step 2**: Use in `ZoneEvaluationHandler` or create custom handler.

---

## Benefits of the Plugin Architecture

### For Developers

- **Minimal code changes**: Add new methods without touching existing code
- **Clear interfaces**: Base classes define exactly what to implement
- **Consistent integration**: All methods work with the same data structures
- **Easy testing**: Test new methods in isolation
- **Version control**: New methods in separate directories reduce conflicts

### For Researchers

- **Algorithm comparison**: Easily test different approaches
- **Reproducibility**: Switch methods via config files
- **Benchmarking**: Fair comparison using same pipeline
- **Rapid prototyping**: Focus on algorithm, not integration

### For Users

- **Flexibility**: Choose best method for your scenario
- **No recompilation**: Change behavior via config
- **Extensibility**: Add custom methods for specific needs
- **Maintainability**: Updates to one method don't affect others

---

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

---

## Dependencies

- **NumPy**: Numerical operations
- **OpenCV (cv2)**: Image processing
- **Matplotlib**: Visualization
- **Shapely**: Geometric operations
- **OR-Tools**: Optimization (MILP for sequencing)
- **PyYAML**: Configuration loading
- **Colorama**: Colored logging

---

## Usage Examples

```python
# Command line
python main.py Map2

# Programmatic
from main import run_pipeline
run_pipeline("Map2")
```

---

## References

- Boustrophedon decomposition: Classical coverage path planning
- Voronoi decomposition: Optimal space partitioning
- Distance Transform: Contour-based path generation
- Plugin Architecture: Extensible software design patterns

---
