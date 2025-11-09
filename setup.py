# setup.py file
from setuptools import setup, find_packages
from pathlib import Path

# Read the project's long description from README or documentation
this_directory = Path(__file__).parent
long_description = ""
readme_file = this_directory / "README.md"
doc_file = this_directory / "documentation.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
elif doc_file.exists():
    long_description = doc_file.read_text(encoding='utf-8')

setup(
    name="covpp",
    version="0.1.0",
    author="MortezaHagh",
    description="Coverage Path Planning system for robotic applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MortezaHagh/coverage-path-planning",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "Shapely>=1.8.0",
        "scipy>=1.7.0",
        "ortools>=9.0.0",
        "pyastar2d>=1.0.0",
        "scikit-image>=0.18.0",
        "colorama>=0.4.0",
        "PyYAML>=5.4.0",
        "pynput>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "covpp=covpp.main:main",
        ],
        "segmentors": [
            "simple=covpp.segmentation.simple_segmentation.segmentation_simple:Segmentation",
            "voronoi=covpp.segmentation.voronoi_segmentation.segmentation_voronoi:Segmentation",
        ],
        "planners": [
            "boust=covpp.planning.boustrophedon.coverage_pp_boust:CoveragePathPlanner",
            "contour=covpp.planning.contour.coverage_pp_contour:CoveragePathPlanner",
        ],
        "sequencers": [
            "dfs=covpp.sequence.sequence_optimizers:DfsSO",
            "milp=covpp.sequence.sequence_optimizers:MilpSO",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
