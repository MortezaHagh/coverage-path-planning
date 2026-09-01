# Sequencing Module

Determines cell visitation order and connects coverage paths into a single trajectory.

## Components

- **SequenceHandler** (`sequence_handler.py`): Orchestrates sequencing and path connection
- **SeqOptimizerBase** (`sequence_optimizer_base.py`): Base class for sequencing optimizers
- **ConnectPaths** (`connect_paths.py`): Generates collision-aware connecting paths between cells
- **CellsGraph** (`cells_graph_1.py`): Builds connectivity graph between cells

## Available Methods

- **dfs**: Depth-first traversal
- **milp**: OR-Tools MILP TSP solver
- **simple**: Nearest-neighbor greedy heuristic

## Data Flow

```
PCell[] (cells with coverage paths)
  ↓
CellsGraph (build distance matrix)
  ↓
SequenceHandler (load optimizer)
  ↓
SeqOptimizerPlugin.create_tour(cost_matrix)
  ↓
cell_order[] (traversal sequence)
  ↓
ConnectPaths (generate connecting paths)
  ↓
final_path (complete trajectory)
  ↓
ZoneEvaluationHandler (compute metrics)
```

## Configuration

Select in `configs.py`:

```python
pipeline_config.sequence_method = "dfs"  # or "milp", "simple"
```

## Path Connection

After determining cell order, `ConnectPaths` generates:
- Collision-aware paths between cell endpoints (A* via `PathPlannerBase`)
- Paths in both pixel and meter space
- Full combined trajectory
