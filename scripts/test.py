
# neighbh_mask_size = 3
# robot_r_px = int(np.ceil(0.5 * self.scfg.robot_circumscribed_r / self.map_data.up_resolution))
# f_map = copy.copy(map_i)
# ------------------------------------------------------------------------------


# cropping ---------------------------------------------------------------------
# # outside mask
# cropped_mask = np.zeros_like(map_cropped)
# # fill mask with white pixels
# cv.fillConvexPoly(cropped_mask, cropped_vertices, 255)
# # replace pixel outside of the zone with black pixels
# for hp in range(0, h_max - h_min):
#     for wp in range(0, w_max - w_min):
#         if cropped_mask[hp, wp] == 0:
#             map_cropped[hp, wp] = 0
#             # never happens!

# # visualize
# visualize_map(map_cropped, "map_cropped")
# visualize_map(cropped_mask)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# def narrow_passage_filter(self, map_i: np.ndarray) -> np.ndarray:
#     """ remove area behind narrow passages based on the robot size.
#     Args:
#         map_i (np.ndarray): input map.
#     Returns:
#         np.ndarray: filtered map based on narrow passages.
#     """
#     print(f"[{self.ns}]: applying filter: narrow passage.")
#     robot_d_px = int(np.ceil(self.scfg.robot_circumscribed_d / self.map_data.up_resolution))
#     f_map = copy.copy(map_i)

#     # Find contours in the map
#     contours, _ = cv.findContours(f_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         # Create a mask for the current contour
#         mask = np.zeros_like(f_map)
#         cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

#         # Distance transform to find narrow passages
#         dist_transform = cv.distanceTransform(mask, cv.DIST_L2, 5)
#         narrow_passages = dist_transform < robot_d_px

#         # Block narrow passages
#         f_map[narrow_passages] = 0

#     return f_map
# ------------------------------------------------------------------------------


# # local max method -----------------------------------------------------------
# {# Define the footprint (neighborhood size) for local maxima detection
# footprint = np.ones((neighbh_mask_size, neighbh_mask_size), dtype=np.uint8)

# # Apply a maximum filter and compare with the original distance map to find local maxima
# max_filtered = maximum_filter(distance_map, footprint=footprint)
# local_max = max_filtered == distance_map

# # Mask out non-local maxima
# local_max_points = np.zeros_like(dist_trans, dtype=np.uint8)
# local_max_points[local_max] = dist_trans[local_max]
# visualize_map(local_max_points, "local_max_points")}

# # Step 4: Optional - Threshold to keep only significant maxima
# threshold_value = np.max(dist_trans) * 0.5  # Keep peaks above 50% of max
# significant_maxima = (local_max_points >= threshold_value).astype(np.uint8)
# visualize_map(significant_maxima, "significant_maxima")
# ------------------------------------------------------------------------------


# gradient ---------------------------------------------------------------------
# {# Compute gradients using Sobel filter
# grad_x = cv.Sobel(dist_trans, cv.CV_8U, 1, 0, ksize=3)  # Gradient in X
# grad_y = cv.Sobel(dist_trans, cv.CV_8U, 0, 1, ksize=3)  # Gradient in Y

# # Compute magnitude and direction of the gradient
# magnitude = cv.magnitude(grad_x, grad_y)
# direction = cv.phase(grad_x, grad_y, angleInDegrees=True)
# visualize_map(magnitude, "magnitude")

# # Apply the Laplacian operator
# dist_trans_float = distance_map.astype(np.uint8)
# laplacian = cv.Laplacian(dist_trans_float, cv.CV_64F)
# print("laplacian", laplacian)
# visualize_map(laplacian, "laplacian")
# norm_lap = cv.normalize(laplacian, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# print("norm_lap", norm_lap)
# visualize_map(norm_lap, "norm_lap")

# min_lap = np.amax(norm_lap)
# test = np.zeros_like(norm_lap)
# test[norm_lap == min_lap] = 255
# visualize_map(test, "test")
# # ----------------------------------------------------------------------------

# # path of points with distance equal to x ------------------------------------
# # Get cells with distance equal to free_space_obst_dist_lim_px and make them red
# dist_trans_rgb = cv.cvtColor(dist_trans, cv.COLOR_GRAY2BGR)
# red_cells = dist_trans == free_space_obst_dist_lim_px
# dist_trans_rgb[red_cells] = [0, 0, 255]
# visualize_map(dist_trans_rgb, "dist_trans_rgb_red")
# # ----------------------------------------------------------------------------


# # display contours -----------------------------------------------------------
# color_image = cv.cvtColor(c_map, cv.COLOR_GRAY2BGR)
# cv.drawContours(color_image, contours, -1, (0, 0, 255), thickness=2)
# cv.imshow("Contours in Red", color_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # ----------------------------------------------------------------------------
