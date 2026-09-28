# Segmentation Improvement

Post-processing module for refining segmentation results.

<!-- todo doc -->

## Operations

- **Boundary smoothing**: Reduces noise from algorithm artifacts
- **Cell merging**: Combines adjacent small cells
- **Area filtering**: Removes cells below minimum size thresholds
- **Boundary adjustment**: Expands or contracts cell boundaries

## Parameters

- **min_area**: Minimum cell area (m²) before merging
- **merge_threshold**: Area ratio threshold for merging adjacent cells
- **smooth_kernel**: Window size for boundary smoothing
- **dilation_steps**: Number of dilation passes for boundary expansion

## References

- Post-processing techniques in computational geometry
- Morphological operations for boundary refinement
