# Coverage Path Planning Pipeline Documentation

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

## Quick Summary

`covpp` is a modular, plugin-driven coverage path planning pipeline for (grid / occupancy) maps. Starting from a grayscale map + YAML, it selects a district, analyzes free space, segments into cells, plans per–cell coverage paths, sequences them, and evaluates performance. Algorithms for segmentation, coverage planning, low-level path finding, and sequencing are loaded via Python entry points (`segmentors`, `coverage_planners`, `path_planners`, `sequencers`) so you can add new methods based on base interfaces without touching core code.

## Overview

This is a comprehensive coverage path planning system that generates optimal paths for robotic coverage of different environments. The pipeline processes map images, performs space segmentation, plans coverage paths, and optimizes the sequence of path execution.

## Pipeline Architecture

The system follows a **modular, plugin-based architecture** where new algorithms can be easily integrated by extending base classes. Each major component (Segmentation, Coverage Planning, Sequence Optimization, path planning, and Evaluation) is designed with abstract base classes that define the interface, allowing developers to add new methods without modifying the core pipeline.

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

### 6. **Zone-Level Coverage Planning** (`ZoneData`) [**Main Planning Step**]

For each zone in the district:

- **Segmentation**: Divides zone into cells
- **Coverage Planning**: Generates paths for each cell
- **Sequencing**: Optimizes cell visit order and connects paths
- **Evaluation**: Computes statistics and metrics

## Module Details

For detailed information about each module, please refer to [Module Details](module_details.md).

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

Saved via `ZoneStatsBase.get_data_dict()`; key fields include:

```json
{
  "map_name": "Map2",
  "method": "boust",
  "zone_ind": 0,
  "trav_time_full": 42.1,
  "trav_time_coverage": 31.0,
  "pt_segmentation": 0.5,
  "pt_coverage_planning": 1.7,
  "pt_sequencing": 0.3,
  "pt_pipeline": 2.5,
  "full_length": 120.55,
  "coverage_length": 95.2,
  "yaw_path_full": 12.33,
  "yaw_path_coverage": 8.76,
  "cleaning_area_px": 215342,
  "area_cleaning_m": 53.84,
  "cleaned_area_px": 210000,
  "area_cleaned_m": 52.5,
  "area_uncleaned_m": 1.34,
  "area_overlaped_m": 4.12,
  "area_coverage_pc": 97.51,
  "area_overlap_pc": 7.65
}
```

District-level aggregation is saved by `DistrictEvaluationHandler` using the per-zone stats.

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
python main.py Map2
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
