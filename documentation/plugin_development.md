# Adding New Methods (Plugin Development)

The plugin-based design makes extending the system straightforward:

## 1. **Adding a New Segmentation Method** (Plugin)

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

## 2. **Adding a New Coverage Planning Method** (Plugin)

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

## 3. **Adding a New Path Planner** (Plugin)

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

## 4. **Adding a New Sequence Optimization Method** (Plugin)

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

## 5. **Adding a New Evaluation Method**

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
