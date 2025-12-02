# Coverage Path Planning (CovPP)

A comprehensive, plugin-based coverage path planning system for robotic applications. This package provides flexible algorithms for decomposing spaces, generating coverage paths, and optimizing traversal sequences for autonomous robots.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Documentation

- **[Full Documentation](documentation/documentation.md)** - **Complete system documentation**
- **[API Documentation](docs/)** - Sphinx-generated API docs

## Pipeline

Map + Data -> Space Analysis -> Generate District -> Create Zones -> Segmentation -> Coverage path planning -> sequence optimization -> evaluation

## Features

- 🔌 **Plugin Architecture**

  - Add new algorithms without modifying core code
  - Configuration-driven method selection
  - Consistent interfaces across components

- 🗺️ **Multiple Segmentation Methods**

  - Simple segmentation (single cell)
  - Voronoi-based decomposition
  - Extensible plugin architecture for custom methods

- 🎯 **Coverage Planning Algorithms**

  - Boustrophedon (back-and-forth sweeping)
  - Contour-following coverage
  - Grid / local-energy minimization sweeps for cluttered rooms
  - Plugin-based design for adding new planners

- 🧭 **Path Planning & Connections**

  - A\* (pyastar2d) based connectors keep cell endpoints feasible
  - `path_planners` entry points make it easy to drop in custom graph searches
  - Shared `PathPlannerBase` handles cost-map generation and coordinate transforms

- 🔄 **Path Sequencing Optimization**

  - Depth-first search (`sequencers: dfs` → `covpp.sequence.dfs_optimizer`)
  - MILP TSP solver (`sequencers: milp` → `covpp.sequence.milp_optimizer`)
  - Optional simple ordering plugin (register `covpp.sequence.simple_optimizer` if needed)
  - `sequencers` entry points make it straightforward to add custom optimizers

- 📊 **Evaluation & Metrics**
  - Path length analysis
  - Coverage percentage
  - Yaw changes and smoothness metrics
  - Performance timing

## Installation

### Quick Install

```bash
# From the repository root (the directory that contains pyproject.toml),
# install in development mode
pip install -e .
```

### Automated Installation

```bash
./install.sh
```

### With Development Tools

```bash
pip install -e ".[dev,docs]"
```

## Quick Start

### Command Line

```bash
# Run coverage planning on a map
python main.py Map2

# Or use the covpp command (after installation)
covpp Map2
```

### Configuration

Configurations are defined in several classes defined in the covpp.base.configs.py file.

Current main config classes:

- PipeLineConfig
- MapConfig
- RobotConfig
- SpaceConfig
- PlanningConfig

You may change them based on your preferences and robot data.

## Project Structure

```text
covpp/
├── covpp/ # Main package
│ ├── base/ # Core utilities and configurations
│ ├── planning/ # Coverage path planning algorithms
│ │ ├── boustrophedon/
│ │ └── contour/
│ ├── segmentation/ # Space decomposition methods
│ │ ├── simple_segmentation/
│ │ └── voronoi_segmentation/
│ ├── sequence/ # Path sequencing optimization
│ ├── evaluation/ # Performance metrics
│ ├── spa/ # Space analysis tools
│ ├── district/ # District and zone management
│ └── gui/ # Interactive tools
├── results/ # Maps and results
├── docs/ # Sphinx documentation
└── main.py # Entry point

```

## Requirements

- Python 3.11 or higher
- NumPy >= 1.20.0
- OpenCV >= 4.5.0
- Matplotlib >= 3.3.0
- Shapely >= 1.8.0
- SciPy >= 1.7.0
- OR-Tools >= 9.0.0
- PyYAML >= 5.4.0

See [requirements.txt](requirements.txt) for the complete list.

## Plugin Architecture

Segmentation, coverage planning, low-level path planners, and sequence optimization modules expose abstract base classes plus entry-point registries (`segmentors`, `planners`, `path_planners`, `sequencers`).
You can develop your own method by subclassing the relevant base class, registering it in `pyproject.toml`, and selecting it via `PipeLineConfig`.

## Results

Results are saved in the `results/<map_name>/District/` directory:

```text
results/Map2/District/
├── map_planning.png  # Visualization
├── result.json       # Statistics
└── Zone_0/
    ├── map_planning.png
    ├── map_segmentation.png
    └── result.json
```

## Contributing

Contributions are welcome! The plugin architecture makes it easy to add new methods:

1. Create your method class inheriting from the appropriate base class
2. Register it with the handler/registry
3. Update configuration to use your method

See [documentation.md](documentation.md) for detailed architecture information.

## Testing

```bash
# Run tests (if available)
pytest tests/

# Run on example maps
python main.py Map0
python main.py Map1
python main.py Map2
```

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{covpp2024,
  author = {MortezaHagh},
  title = {Coverage Path Planning System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MortezaHagh/coverage-path-planning}
}
```

## Acknowledgments

- OR-Tools for optimization algorithms
- OpenCV for image processing
- Shapely for geometric operations

## Contact

- **Author**: MortezaHagh
- **Repository**: [https://github.com/MortezaHagh/coverage-path-planning](https://github.com/MortezaHagh/coverage-path-planning)
- **Issues**: [GitHub Issues](https://github.com/MortezaHagh/coverage-path-planning/issues)

## Roadmap

- [ ] Implement additional planners (spiral, random)
- [ ] Path tracking module
- [ ] Add more segmentation methods (grid-based, skeleton-based)
- [ ] Performance benchmarking suite
- [ ] ROS integration
- [ ] Web-based visualization
- [ ] Multi-robot support
