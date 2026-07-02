# Development Instructions

## Comments and docstrings
- Use docstrings for all public classes and methods, following the Google style guide.
- Use comments for non-obvious code, especially for algorithms and data flow.
- Each class and method should have a docstring describing its purpose, what it calculates, important parameters and outputs.
- For functions which include algorithms, provide a description of the algorithm and its steps, but avoid repeating the code itself (like a pseudocode).

## Code structure and naming
- Follow the existing package structure and naming conventions.

## Coding improvements
- check and update docstrings to reflect the current implementation.
- Check and update the input output types in type hints. 

## Credit-saving workflow

- Start from the nearest owning file, handler, or base class.
- Use the existing project docs first for package layout, output paths, and plugin names.
- Ask for the smallest needed scope when the task is ambiguous.
- Avoid repo-wide search unless a local check fails or the behavior is genuinely cross-cutting.
- After a change, run the cheapest focused validation that can falsify the current assumption.