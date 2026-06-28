# Main Context

CovPP is a plugin-driven coverage path planning pipeline for occupancy-grid maps. It loads a map, prepares a safe workspace, creates a district, decomposes each zone into cells, plans coverage paths, sequences those paths, and evaluates the result.

## Main Flow

1. `covpp.main` loads inputs and configuration.
2. `MapImageProcessor` converts the raw map into processed map data.
3. `DistrictGen` creates or selects the district rectangle.
4. `SpaceAnalyzer` builds the workspace and zone map data.
5. `DistrictData` creates zones and runs the zone pipeline.
6. Each `ZoneData` runs segmentation, coverage planning, sequencing, and evaluation.

## Package Structure

- `covpp/base/`: configuration dataclasses, image handling, geometry, logging, and shared helpers
- `covpp/district/`: district creation, zone management, and district-level aggregation
- `covpp/spa/`: workspace analysis and rectangle helpers
- `covpp/segmentation/`: cell decomposition and merge logic
- `covpp/coverage_planning/`: coverage planners for each cell
- `covpp/path_planning/`: local path planners used to connect paths
- `covpp/sequence/`: path ordering and connection logic
- `covpp/evaluation/`: zone and district metrics
- `covpp/gui/`: interactive rectangle selection and drawing helpers

## Important Files

- `pyproject.toml`: package metadata, dependencies, script entry point, and plugin registries
- `setup.py`: legacy setuptools wrapper for editable installs
- `README.md`: short project overview and quick start
- `documentation/`: detailed architecture and module references
- `instructions/`: concise guidance for AI-assisted edits

## Notes

- The package uses the `covpp` console script, so most runs should start from `covpp <map_name>` after installation.
- Result artifacts are written into the `maps/` tree, with district and zone folders created per run.
