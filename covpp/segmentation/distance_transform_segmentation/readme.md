# Distance Transform Segmentation

## Main idea

Instead of erosion, compute a distance transform.

Threshold the distance map repeatedly.

The threshold producing the largest number of room centers is selected.

Then grow these centers to cover the map.

## Inputs

Binary occupancy map

## Pseudocode

```pseudocode
FUNCTION DistanceTransformSegmentation(binary_map):

    distance ← DistanceTransform(binary_map)

    best_threshold ← None
    best_centers ← ∅
    max_center_count ← 0

    max_distance ← Maximum(distance)

    -------------------------------------
    // Search threshold
    -------------------------------------

    FOR threshold
        FROM max_distance
        DOWN TO 1:

        mask ← distance >= threshold

        centers ← ConnectedComponents(mask)

        IF
            NumberOfComponents(centers)
            >
            max_center_count:

            max_center_count ← NumberOfComponents(centers)

            best_threshold ← threshold

            best_centers ← centers

    END FOR

    -------------------------------------
    // Initialize labels
    -------------------------------------

    Label every center uniquely

    -------------------------------------
    // Wavefront expansion
    -------------------------------------

    Queue ← labeled pixels

    WHILE Queue not empty:

        p ← pop()

        FOR neighbors:

            IF
                free
                AND unlabeled:

                    assign label

                    push

    RETURN label_map
```

## Complexity

Distance transform

O(N)

Threshold search

O(TN)

T = number of thresholds

Usually fast because

T << image size
