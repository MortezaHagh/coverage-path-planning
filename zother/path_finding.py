from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

matrix = [
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1]
]
grid = Grid(matrix=matrix)

start = grid.node(0, 0)
end = grid.node(3, 2)

finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
path, runs = finder.find_path(start, end, grid)

print('operations:', runs, 'path length:', len(path))
print(grid.grid_str(path=path, start=start, end=end))

path_list = [(x, y) for x, y in path]
print("Path:", path_list)


# ============================================================================================

# #
# import cv2


# image = "/workspaces/covpp/maps/Map2/AMain/merge.png"

# # load image

# img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
# # plt.imshow(img, cmap='gray')
# # plt.show()

# # create matrix for pathfinding
# FREE = 200
# OCC = 0
# cost_map = np.where(img > FREE, 1, 0)
# # plt.imshow(cost_map, cmap='gray')
# # plt.show()


# # cost_map size
# h, w = cost_map.shape
# print("Cost map size:", w, h)


# # start - end
# start = (50, 50)
# end = (100, 100)

# # map values at start - end
# print("Start value:", img[start[1], start[0]])  # swap x y
# print("End value:", img[end[1], end[0]])  # swap x y

# # swap x y to match the cost map
# start_s = (start[1], start[0])
# end_s = (end[1], end[0])

# # find path using pathfinding library

# grid = Grid(matrix=cost_map)
# start_node = grid.node(*start_s)
# end_node = grid.node(*end_s)

# finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
# path, _ = finder.find_path(start_node, end_node, grid)

# print('Path length:', len(path))

# # plot path
# path_px = [(x, y) for x, y in path]
# # plot
# for p in path_px:
#     print(p)
#     img[p[1], p[0]] = 128  # swap x y

# plt.imshow(img, cmap='gray')
# plt.show()
