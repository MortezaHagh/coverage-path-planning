# Plugin Development

CovPP uses entry points plus base classes so new methods can be added without changing the handlers.

## Segmentation Plugin

1. Create a module under `covpp/segmentation/<your_method>/`.
2. Subclass `SegmentationBase` and implement `run()`.
3. Register the class under the `segmentors` entry-point group in `pyproject.toml`.
4. Set `PipeLineConfig.segmentation_method` to the new name.

Example:

```python
from covpp.segmentation.segmentation_base import SegmentationBase


class Segmentation(SegmentationBase):
    def run(self):
        self.cells = []
        self.n_cells = 0
```

## Coverage Planner Plugin

1. Create a module under `covpp/coverage_planning/<your_method>/`.
2. Subclass `CoveragePathPlannerBase` and implement `coverage_planning()`.
3. Register the class under `coverage_planners`.
4. Set `PipeLineConfig.covpp_method` to the new name.

## Path Planner Plugin

1. Create a module under `covpp/path_planning/<your_method>/`.
2. Subclass `PathPlannerBase` and implement `plan_path(start, end)`.
3. Register the class under `path_planners`.
4. Set `PipeLineConfig.pp_method` to the new name.

## Sequencer Plugin

1. Create a module under `covpp/sequence/<your_method>/`.
2. Subclass `SeqOptimizerBase` and implement `create_tour()`.
3. Register the class under `sequencers`.
4. Set `PipeLineConfig.sequence_method` to the new name.

## Evaluation Plugin

1. Subclass `ZoneEvaluationBase` in `covpp/evaluation/`.
2. Implement `evaluate()` and populate `self.stats`.
3. Use the new class from `ZoneEvaluationHandler` or a custom handler.

## Notes

- Keep plugin names short and stable.
- Match class names used by the existing handlers.
- Update the docs and configuration examples whenever a new plugin group member is added.
