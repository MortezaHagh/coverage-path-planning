# Data

## How Planning Results Are Stored

Each map has its own directory under `maps/`.

The expected layout is:

```text
maps/
    map_name/
        AMain/
        District/
            result.json
            mp_<method>.png
            Zone_0/
                segmentation/
                <method>/
                result.json
            Zone_1/
                segmentation/
                <method>/
                result.json
```

## What Each Folder Contains

- `AMain/`: original map files and map metadata.
- `District/`: district-level outputs and the final merged planning map.
- `Zone_<i>/segmentation/`: segmentation artifacts for the zone.
- `Zone_<i>/<method>/`: method-specific outputs for the zone.
- `result.json`: serialized zone or district statistics.

## Method Names

The full method name is built from the configured segmentation, coverage planning, and sequence methods.

Example:

- `skeleton_contour_milp`
- `voronoi_boust_dfs`

The method name is used in file names and output folders so every run stays isolated.
