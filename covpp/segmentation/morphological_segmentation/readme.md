# Morphological Segmentation

## Main idea

Repeatedly erode free space.

Whenever erosion disconnects a sufficiently large region, record it as a room.

Finally grow all rooms back using wavefront propagation.

## Inputs

Binary occupancy map
1 = free
0 = obstacle

## Parameters

erosion_kernel = 3x3
min_room_area(step)
max_room_area(step)

The thresholds depend on erosion depth.

## Pseudocode

FUNCTION MorphologicalSegmentation(binary_map):

    M1 ← copy(binary_map)          // working map
    M2 ← empty label map

    room_id ← 1
    erosion_step ← 0

    WHILE free cells still exist in M1:

        erosion_step += 1

        -------------------------------------
        // Erode accessible space
        -------------------------------------

        M1 ← MorphologicalErosion(M1)

        -------------------------------------
        // Connected components
        -------------------------------------

        components ← ConnectedComponents(M1)

        FOR each component C:

            area ← NumberOfPixels(C)

            IF
                area > LowerThreshold(erosion_step)
                AND
                area < UpperThreshold(erosion_step):

                    Label C in M2 using room_id

                    Remove C from M1
                    room_id += 1

        END FOR

    END WHILE

    -------------------------------------
    // Grow labels back
    -------------------------------------

    Queue ← all labeled pixels

    WHILE Queue not empty:

        p ← pop(Queue)

        FOR each 4-neighbor n:

            IF
                n is free
                AND n unlabeled:

                    label(n)=label(p)

                    push(n)

        END FOR

    END WHILE

    RETURN M2

## Complexity

Connected components:
O(N)

Each erosion:
O(N)

k erosion iterations

Total

O(kN)
