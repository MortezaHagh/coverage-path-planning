# Boustrophedon Coverage Planning

Back-and-forth parallel sweeping coverage path planner.

## Algorithm

1. Determine dominant sweep direction (based on cell orientation)
2. Generate parallel sweep lines at coverage spacing
3. Identify intersections of sweep lines with cell boundaries
4. Order sweep lines to form a connected path
5. Create waypoints along each sweep line
6. Connect sweep lines to form a continuous path

## Input

- Cell polygon (boundary and obstacles)
- Robot configuration
- Coverage spacing parameter

## Output

`MyPath` object containing:
- Ordered waypoints visiting all sweep lines
- Coverage direction
- Path metrics

## Parameters

- **coverage_spacing**: Distance between parallel sweep lines (meters)

## Configuration

Select in `PipeLineConfig`:

```python
pipeline_config.covpp_method = "boust"
pipeline_config.coverage_spacing = 0.5  # meters between sweep lines
```

## References

- Classical coverage path planning technique
- Well-documented in robotics literature
- Baseline algorithm for many CPP problems
