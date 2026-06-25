# Data

## How planning results are stored

In maps directory, there is a folder for each map name.
It should initially containd a folder called Amain which contains map meta data includeing:
- todo ...

When running, based on pipeline configs, a complete method name is generated, like `skeleton_contour_milp`.
After running the pipeline, for each zone a new folder is created, like zone_0, zone_1, ...
Inside each zone folder, all results of the computation and simulation are stored under the methods name. Also a folder named `segmentation` is created which contains the segmentation results for that zone.
Also, the final planning map with the methods name is created and saved under the map name folder, like `skeleton_contour_milp.png`. This is a visual representation of the final planning results.

So the final structure of the map folder is like this:

```text
maps/
    map_name/
        Amain/
        zone_0/
            segmentation/
            skeleton_contour_milp/
        zone_1/
            segmentation/
            skeleton_contour_milp/
        zone_2/
            segmentation/
            skeleton_contour_milp/
        skeleton_contour_milp.png
```
