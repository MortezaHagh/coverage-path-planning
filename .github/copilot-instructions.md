# Package Context and Instructions

CovPP is a modular coverage path planning project for occupancy-grid maps. The main entry point is `covpp.main:main`, exposed as the `covpp` console script.

## Project Structure

- `covpp/`: core package code
- `covpp/base/`: configs, geometry, image processing, logging, and shared utilities
- `covpp/district/`: district creation, zone orchestration, and map aggregation
- `covpp/spa/`: space analysis and workspace geometry helpers
- `covpp/segmentation/`: zone decomposition into planning cells
- `covpp/coverage_planning/`: per-cell coverage path planners
- `covpp/path_planning/`: low-level path planners used to connect cells
- `covpp/sequence/`: sequencing and path-connection logic
- `covpp/evaluation/`: zone and district metrics
- `covpp/gui/`: interactive rectangle selection and drawing helpers
- `maps/`: input maps and generated results
- `documentation/`: project documentation
- `instructions/`: workspace guidance for AI and contributors

## Working Rules

- Keep edits focused on the smallest code path that controls the behavior.
- Use the project terminology: district, zone, segmentation, coverage planning, sequencing, evaluation.
- Prefer existing handlers and base classes over ad hoc logic.
- Keep comments and documentation concise, but explain non-obvious algorithms and data flow.
- Do not invent module names, entry points, or directories. Match the current package layout and `pyproject.toml`.

## Credit-Saving Rules

- Check the local docs first before broad code exploration: `instructions/MAIN_CONTEXT.md`, `documentation/data.md`, and `documentation/module_details.md`.
- Prefer narrow, local searches around the owning handler or base class instead of scanning the full repository.
- Make one small edit and then run one focused validation step before widening scope.
- When the question is about package structure or result data, answer from the docs unless the code has clearly diverged.
- Avoid creating extra files unless they will be reused often; a short instruction update is usually cheaper than a new agent.

## Entry Points

- `segmentors`: `simple`, `voronoi`, `skeleton`
- `coverage_planners`: `boust`, `contour`, `grid_energy_min`
- `path_planners`: `astar`
- `sequencers`: `dfs`, `milp`

## Output Layout

- Map metadata lives under `maps/<map_name>/AMain/`
- District outputs live under `maps/<map_name>/District/`
- Zone outputs live under `maps/<map_name>/District/Zone_<i>/`
- Each zone stores segmentation results, method-specific artifacts, and `result.json`

## Documentation Style

- Keep prose short and direct.
- Prefer markdown tables for module or plugin overviews.
- Use the actual class and module names from the package.
- Update docs when configuration, entry points, or output paths change.
