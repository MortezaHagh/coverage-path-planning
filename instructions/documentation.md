# Documentation

Use this workspace to keep AI-facing notes and project documentation aligned with the actual codebase.

## What To Document

- The end-to-end pipeline from map input to evaluation output.
- Module responsibilities and the main classes in each package.
- Plugin registries and the exact entry-point groups in `pyproject.toml`.
- Result folder layout under `maps/`.
- Configuration keys that affect planning behavior.

## Writing Rules

- Keep explanations concise and concrete.
- Use current names from the package, not legacy examples.
- Prefer small tables or short sections over long narrative blocks.
- When documenting examples, match the real directory names and method names in the code.
- For algorithms and some complex functions, include the process description and details of the implementation, but avoid repeating the code itself.
- Keep AI-facing docs short enough that they can answer common package/data questions without extra repository scanning.

## Update Triggers

- Update docs when a plugin group changes.
- Update docs when output paths or folder names change.
- Update docs when the pipeline order changes.
- Update docs when a new public module or major class is added.
