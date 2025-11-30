"""Legacy setuptools entry point.

All packaging metadata now lives in pyproject.toml, but keeping this file
allows editable installs via ``pip install -e .`` on older tooling.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
