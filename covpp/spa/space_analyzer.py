"""Space Analyzer Module

Analyze space to generate local free-space maps and zone-level MapData
based on robot and workspace configurations.

Steps:
- Crop and frame the district/zone rectangle
- Derive workspace (ws) and cleaning maps via dilation/erosion or distance maps
- Select a connected free-space island (optionally via user click)
- Remove islands and compute final free-space bounding rectangle
- Build visualization layers and assemble a `ZoneMapData`

Result:
- Populated maps (orig/ws/cleaning/selected) and a ready-to-use `ZoneMapData`.
"""
import copy
import cv2 as cv
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.map_data import MapData
from covpp.gui.rectangle import Rectangle
from covpp.base.configs import SpaceConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.spa.spa_rectangles import find_circumscribed_rectangle
# from covpp.spa.inscribed_rectangle_histogram import find_inscribed_rotated_rectangle
FREE_COST = 255
OCCUPIED_COST = 0


class SpaceAnalyzer:
    """Space analysis pipeline for free-space map creation.

    Notes on map types:
    - ws: workspace (feasible robot-center positions)
    - cleaning: cleanable area given the robot radius
    - orig: original or eroded-back after dilation
    - sf: selected free-space (single island)
    - f: filtered (island removal)

    Key methods:
    - ``create_local_map_ws_selection``: Build ws/cleaning maps, select island, compute artifacts
    - ``update_by_rectangle``: Re-run analysis for a new zone rectangle
    - ``create_zone_map_data``: Package outputs into a `ZoneMapData`
    """

    def __init__(self, map_data: MapData, rectangle: Rectangle):
        self.lg = MyLogger("SpaceAnalyzer")
        self.lg.info1("Step: Analyzing space for creating free space [local map]")

        # configs
        self.scfg: SpaceConfig = SpaceConfig()
        self.space_type: str = "District"  # District, Zone
        self.simple: bool = True
        self.plot_main: bool = True

        # inputs
        self.orig_map_data: MapData = map_data          # original input map data - fixed
        self.orig_rectangle: Rectangle = rectangle      # Initial district rectangle - won't be changed
        self.rectangle: Rectangle = rectangle           # Current spa rectangle (updated) for each zone
        self.rect_data: dict = rectangle.data           # Current rectangle data (updated) keys = {"xc", "yc", "rot", "w", "h"}
        self.vertices: np.ndarray = np.array(rectangle.vertices_t, dtype=np.int32)  # rectangle vertices - (updated) for each zone
        self.binary_map_0: np.ndarray = map_data.up_bin_map     # original input binary map - fixed

        # attributes
        self.hw_bd: dict[str:int] = None                    # map hw bound dict keys = {"h_min", "w_min", "h_max", "w_max"}
        self.hw_bd_local: dict[str:int] = None              # local hw bound dict keys = {"h_min", "w_min", "h_max", "w_max"}
        self.final_rect_data: dict = None                   # free space bounding box rectangle data
        self.final_vertices: list[tuple[int, int]] = None   # free space bounding box vertices
        #
        self.is_fs_point_selected: bool = False
        self.fs_selected_point: tuple[float, float] = None

        # zone map data - created after space analysis
        self.__zone_map_data: ZoneMapData = None

        # # maps
        # orig: without dilation/erosion
        # ws: work space
        # cleaning: cleanable area
        # sf: selected free space
        # f: filtered (island detection)
        self.cropped_mask: np.ndarray = None
        self.map_c_orig: np.ndarray = None                  # cropped original map based on rectangle
        self.map_1_dilated_cleaning: np.ndarray = None      # white area: cleanable area
        self.map_2_dilated_ws: np.ndarray = None            # white area: work space area - robot center feasible area
        self.map_3_c_dilated_cleaning: np.ndarray = None    # cropped cleaning map
        self.map_4_c_dilated_ws: np.ndarray = None          # cropped work space map
        self.map_6_ws_sf_f: np.ndarray = None               # work space selected free space filtered
        self.map_7_c_ws_sf_f: np.ndarray = None             # cropped work space selected free space filtered
        self.map_8_ws_sf_f_orig: np.ndarray = None          # work space selected free space filtered original
        self.map_9_ws_sf_f_cleaning: np.ndarray = None      # work space selected free space filtered cleaning
        self.map_10_c_ws_sf_f_orig: np.ndarray = None       # cropped work space selected free space filtered original
        self.map_11_c_ws_sf_f_cleaning: np.ndarray = None   # cropped work space selected free space filtered cleaning
        self.map_viz: np.ndarray = None                     # final map visualization

        # process
        self.calculate_hw_bounds()

    @property
    def zone_map_data(self) -> ZoneMapData:
        """ Return zone map data object after space analysis. """
        return copy.deepcopy(self.__zone_map_data)

    # ==========================================================================

    def update_by_rectangle(self, rectangle: Rectangle) -> None:
        """Recompute maps for a new zone rectangle.

        Args:
        - rectangle: New zone frame to apply for local analysis.

        Steps:
        - Update internal rectangle, vertices, and bounds
        - Re-crop all derived maps to the new frame
        - Recompute final free-space bounding rectangle and visualization
        - Refresh the `ZoneMapData`

        Returns:
        - None
        """
        self.space_type = "Zone"
        self.rectangle = rectangle
        self.vertices = np.array(rectangle.vertices_t, dtype=np.int32)
        self.rect_data = rectangle.data
        self.calculate_hw_bounds()
        #
        self.map_c_orig = self.crop_and_frame(self.binary_map_0)
        self.map_3_c_dilated_cleaning = self.crop_and_frame(self.map_1_dilated_cleaning)
        self.map_4_c_dilated_ws = self.crop_and_frame(self.map_2_dilated_ws)
        self.map_7_c_ws_sf_f = self.crop_and_frame(self.map_6_ws_sf_f)
        self.map_10_c_ws_sf_f_orig = self.crop_and_frame(self.map_8_ws_sf_f_orig)
        self.map_11_c_ws_sf_f_cleaning = self.crop_and_frame(self.map_9_ws_sf_f_cleaning)

        # free space bounding box
        # rect = self.get_final_freespace_bd_rect(self.map_11_c_ws_sf_f_cleaning)
        rect = self.get_final_freespace_bd_rect(self.map_7_c_ws_sf_f)
        self.final_rect_data = rect["rect_data"]    # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self.final_vertices = rect["vertices"]      # vertices: [(x, y), ...]

        # final map visualization
        self.map_viz = self.create_final_vis_map()

        # create district object
        self.create_zone_map_data()

    def create_local_map_ws_selection(self) -> None:
        """Build local ws/cleaning maps and select a free-space island.

        Steps:
        - Crop input binary map to the district rectangle
        - Compute cleaning and workspace maps (dilation/erosion or distance)
        - Select a single free-space island via click or center default
        - Remove islands to ensure accessibility
        - Derive cleaning and original maps for the selected island
        - Compute final free-space bounding rectangle
        - Create final visualization and package data

        Returns:
        - None
        """

        # crop map and mask
        binary_map = self.binary_map_0
        self.map_c_orig = self.crop_map(binary_map)
        self.cropped_mask = np.ones_like(self.map_c_orig) * FREE_COST

        # detect workspace (based on robot size and distance conditions)
        if self.simple:
            # simple version - dilate obstacles
            map_1_dilated_cleaning = self.dilate_erode_map(map_i=binary_map, radius_m=self.scfg.uncleaned_dist, erode=True)
            map_2_dilated_ws = self.dilate_erode_map(map_i=binary_map, radius_m=self.scfg.ws_r_dist, erode=True)
            map_3_c_dilated_cleaning = self.crop_map(map_1_dilated_cleaning)
            map_4_c_dilated_ws = self.crop_map(map_2_dilated_ws)
        else:
            # apply distance conditions using distance transform
            map_3_c_dilated_cleaning = self.distance_filter(map_full=binary_map)
            map_4_c_dilated_ws = self.distance_filter(map_full=binary_map)
        self.map_1_dilated_cleaning = map_1_dilated_cleaning
        self.map_3_c_dilated_cleaning = map_3_c_dilated_cleaning
        self.map_2_dilated_ws = map_2_dilated_ws
        self.map_4_c_dilated_ws = map_4_c_dilated_ws

        # select free space point
        self.select_free_space(map_4_c_dilated_ws)
        map_5_ws_sf = self.separate_selected_free_space(map_2_dilated_ws)  # selected work space

        # remove islands
        map_6_ws_sf_f = self.remove_islands(map_5_ws_sf)
        self.map_6_ws_sf_f = map_6_ws_sf_f
        self.map_7_c_ws_sf_f = self.crop_map(self.map_6_ws_sf_f)

        # get cleaning space
        self.map_8_ws_sf_f_orig = self.dilate_erode_map(map_i=map_6_ws_sf_f, radius_m=self.scfg.ws_r_dist, erode=False)
        self.map_9_ws_sf_f_cleaning = self.dilate_erode_map(map_i=map_6_ws_sf_f, radius_m=self.scfg.cleaning_r, erode=False)

        # maps
        self.map_10_c_ws_sf_f_orig = self.crop_map(self.map_8_ws_sf_f_orig)
        self.map_11_c_ws_sf_f_cleaning = self.crop_map(self.map_9_ws_sf_f_cleaning)

        # free space bounding box
        rect = self.get_final_freespace_bd_rect(self.map_10_c_ws_sf_f_orig)
        self.final_rect_data = rect["rect_data"]    # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self.final_vertices = rect["vertices"]      # vertices: [(x, y), ...]

        # final map visualization
        self.map_viz = self.create_final_vis_map()

        # create district object
        self.create_zone_map_data()

    def create_zone_map_data(self) -> None:
        """Populate and store a `ZoneMapData` from current artifacts.

        Sets original map data, rectangles, derived maps, visualization, and
        pixel bounds. Keeps an internal deep-copiable snapshot via property
        `zone_map_data`.

        Returns:
        - None
        """
        zone_map_data = ZoneMapData()
        zone_map_data.set_orig_map_data(self.orig_map_data)
        zone_map_data.set_orig_rectangle(self.orig_rectangle)
        zone_map_data.set_rectangle(self.rectangle)
        zone_map_data.set_rect_data(self.rect_data, self.vertices)
        zone_map_data.set_final_rect_data(self.final_rect_data, self.final_vertices)
        zone_map_data.set_map_orig(self.map_c_orig)
        zone_map_data.set_map_ws(self.map_7_c_ws_sf_f)
        zone_map_data.set_map_sf_orig(self.map_10_c_ws_sf_f_orig)
        zone_map_data.set_map_cleaning(self.map_11_c_ws_sf_f_cleaning)
        zone_map_data.set_map_viz(self.map_viz)
        zone_map_data.set_hw_bd_px(self.hw_bd)
        self.__zone_map_data = zone_map_data

    def get_final_freespace_bd_rect(self, map_i: np.ndarray) -> dict:
        """Compute bounding rectangle of selected free space in global frame.

        Args:
        - map_i: Map in local frame used to derive the rectangle.

        Returns:
        - dict: {"vertices": np.ndarray[(N,2)], "rect_data": {"xc","yc","rot","w","h"}}

        Notes:
        - Uses local bounding rectangle then shifts to global coordinates
        - Returns empty dict values if no valid rectangle is found
        """

        final_map = map_i.copy()

        # # dilate
        # radius_px = int(np.ceil(self.scfg.uncleaned_dist / self.orig_map_data.up_resolution))
        # kernel_size = 2*radius_px + 1
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # final_map = cv.dilate(map_i, kernel)
        # final_map = np.where(final_map > 0, 255, 0).astype(np.uint8)

        #
        local_rect = self.find_bounding_rect(final_map)
        # {"vertices": vertices, "rect_data": data}
        # vertices: [(x, y), ...]
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}

        # check validity
        if local_rect is None:
            self.lg.error("[get_final_freespace_bd_rect]: Failed to find a valid rectangle")
            return {"vertices": None, "rect_data": None}

        # local to global rectangle
        vertices = local_rect["vertices"] + np.array([self.hw_bd["w_min"], self.hw_bd["h_min"]])
        data = local_rect["rect_data"]
        data["xc"] += self.hw_bd["w_min"]
        data["yc"] += self.hw_bd["h_min"]
        rect = {"vertices": vertices, "rect_data": local_rect["rect_data"]}
        return rect

    def find_bounding_rect(self, map_i: np.ndarray) -> dict:
        """Find bounding rectangle of the free space in local frame.

        Args:
        - map_i: Local map image with selected free-space pixels.

        Returns:
        - dict: {"vertices": np.ndarray[(N,2)], "rect_data": {"xc","yc","rot","w","h"}}
        """
        if self.space_type == "District":
            sol = find_circumscribed_rectangle(map_i)
        else:
            sol = find_circumscribed_rectangle(map_i)
            # sol = find_inscribed_rotated_rectangle(map_i)
            # sol: {"vertices": vertices, "rect_data": data}
            # vertices: [(x, y), ...]
            # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        return sol

    # ==========================================================================

    def distance_filter(self, map_full: np.ndarray) -> np.ndarray:
        """Detect free spaces using distance transform with obstacle clearance.

        Args:
        - map_full: Full binary map; defaults to `self.binary_map_0` when None.

        Returns:
        - np.ndarray: Cropped workspace map meeting distance-to-obstacle threshold.
        """
        return self.detect_free_spaces_dist_occ(map_full=map_full, dist_from_occ_m=self.scfg.ws_r_dist)

    def detect_free_spaces_dist_occ(self, map_full: np.ndarray, dist_from_occ_m: float = 0.0) -> np.ndarray:
        """Detect free spaces considering distance from obstacles using distance transform.

        Args:
        - map_full: Full binary map with FREE/OCCUPIED labels.
        - dist_from_occ_m: Required clearance from obstacles, in meters.

        Steps:
        - Expand crop to account for border effects
        - Compute L2 distance transform and normalize
        - Threshold based on required clearance and crop back

        Returns:
        - np.ndarray: Cropped workspace map satisfying the distance criterion.
        """

        # current map
        if map_full is None:
            map_full = self.binary_map_0

        # Area for appling distance transform - expanding the cropped map - e -> expanded
        # Additional distance is necessary to check if any obstacle isn't just outside of the zone border
        dist_from_occ_px = int(np.ceil(dist_from_occ_m / self.orig_map_data.up_resolution))
        e_min_obst_dist = dist_from_occ_px * self.scfg.scaler_os_search
        e_hw_l = copy.copy(self.hw_bd)
        e_hw_l["w_min"] -= e_min_obst_dist
        e_hw_l["h_min"] -= e_min_obst_dist
        e_hw_l["w_max"] += e_min_obst_dist
        e_hw_l["h_max"] += e_min_obst_dist
        e_hw_l = self.apply_hw_limits(e_hw_l)
        d_h_min = self.hw_bd["h_min"] - e_hw_l["h_min"]
        d_w_min = self.hw_bd["w_min"] - e_hw_l["w_min"]

        # expanded cropped map for distance calculation
        e_c_map = map_full[e_hw_l["h_min"]:e_hw_l["h_max"], e_hw_l["w_min"]:e_hw_l["w_max"]]

        # computing distance transform for local map (cropped map)
        dist_trans = cv.distanceTransform(e_c_map, distanceType=cv.DIST_L2, maskSize=5, dstType=cv.CV_8U)
        max_dist_px = np.amax(dist_trans)
        dist_trans = cv.normalize(dist_trans, None, alpha=0, beta=FREE_COST, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # free space distance limit
        norm_dist_from_occ_px = np.ceil(dist_from_occ_px * FREE_COST / max_dist_px)

        # eliminate expanded part from dist_trans
        n_h, n_w = self.cropped_mask.shape
        dist_trans = dist_trans[d_h_min:d_h_min+n_h, d_w_min:d_w_min+n_w]

        # free map
        map_ws = copy.copy(self.cropped_mask)
        map_ws[dist_trans < norm_dist_from_occ_px] = OCCUPIED_COST

        # visualize_map(map_ws, "map_ws")
        return map_ws

    def select_free_space(self, map_i: np.ndarray) -> None:
        """Select the seed point for free-space island extraction.

        Args:
        - map_i: Cropped workspace map shown for selection context.

        Behavior:
        - If `scfg.select_free_space` is False, selects map center by default
        - Otherwise opens an interactive window and waits for a click

        Returns:
        - None
        """
        self.lg.info("selecting the free space ...")
        free_map_disp = cv.addWeighted(self.map_c_orig, 0.4, map_i, 0.6, 0)
        if not self.scfg.select_free_space:
            self.is_fs_point_selected = True
            self.fs_selected_point = (free_map_disp.shape[1] // 2, free_map_disp.shape[0] // 2)  # center of the map
            return
        win_name = "select free space"
        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(win_name, self.select_point_cb)
        cv.imshow(win_name, free_map_disp)
        while not self.is_fs_point_selected:
            cv.waitKey(1)
        cv.destroyAllWindows()

    def select_point_cb(self, event, x, y, _, __) -> None:
        """Mouse callback to capture the selected point for island extraction."""
        if event == cv.EVENT_LBUTTONDOWN:
            self.fs_selected_point = (x, y)
            self.is_fs_point_selected = True

    def separate_selected_free_space(self, c_map: np.ndarray) -> np.ndarray:
        """Extract the free-space island containing the selected point.

        Steps:
        - Find external contours
        - Choose the contour containing the selected point, or the largest
        - Fill and mask to isolate the selected island

        Args:
        - c_map: Cropped workspace map.

        Returns:
        - np.ndarray: Binary map of the selected island.
        """
        self.lg.info("selecting the open space based on the selecting point.")

        # get contours
        contours, hierarchy = cv.findContours(c_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # pylint: disable=W0612
        if len(contours) == 0:
            self.lg.info("contours is empty")
            return c_map

        # # display contours
        # self.display_contour(contours)

        # find selected_contour
        self.lg.info3(f"number of contours (free islands): {len(contours)}")
        for contour in contours:
            if cv.pointPolygonTest(contour, self.fs_selected_point, True) >= 0:
                selected_contour = contour
                break
        else:
            # Find largest countour if the selection do not point to any countour
            if len(contours) != 0:
                largest_contour_ind = max(range(len(contours)), key=lambda i: cv.contourArea(contours[i]))
                selected_contour = contours[largest_contour_ind]

        # detect selected_free_space
        selected_free_space = np.zeros_like(c_map)
        cv.drawContours(selected_free_space, [selected_contour], 0, FREE_COST, cv.FILLED)
        map_fs = cv.bitwise_and(c_map, selected_free_space)
        return map_fs

    def remove_islands(self, map_i: np.ndarray) -> np.ndarray:
        """Remove inaccessible islands (holes) from the selected free-space map.

        Returns:
        - np.ndarray: Filtered map with holes removed.
        """
        contours, hierarchy = cv.findContours(map_i, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return map_i

        # remove area of a contour with parent contour
        map_fs_rif = np.ones_like(map_i) * FREE_COST
        for i, element in enumerate(hierarchy[0]):
            if element[3] != -1:
                cv.drawContours(map_fs_rif, [contours[i]], 0, OCCUPIED_COST, cv.FILLED)
                cv.drawContours(map_fs_rif, [contours[i]], 0, FREE_COST, 1)
        map_fs_rif = cv.bitwise_and(map_i, map_fs_rif)
        return map_fs_rif

    # ==========================================================================

    def calculate_hw_bounds(self) -> None:
        """Calculate pixel bounds of the current rectangle.

        Result keys:
        - `hw_bd`: dict with h_min, w_min, h_max, w_max in global frame
        - `hw_bd_local`: dict with local frame bounds starting at (0,0)
        """
        wh_max = np.max(self.vertices, axis=0)
        wh_min = np.min(self.vertices, axis=0)
        w_max, h_max = wh_max.tolist()
        w_min, h_min = wh_min.tolist()
        hw_limits = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        hw_bd = self.apply_hw_limits(hw_limits)
        self.hw_bd = hw_bd
        self.hw_bd_local = {"h_min": 0, "w_min": 0, "h_max": h_max-h_min, "w_max": w_max-w_min}

    def crop_and_frame(self, map_i: np.ndarray) -> np.ndarray:
        """Crop the map to `hw_bd` and draw the rectangle frame on top."""
        map_cropped = self.crop_map(map_i)
        map_framed = self.make_frame(map_cropped, local_frame=True, value=OCCUPIED_COST)
        return map_framed

    def crop_map(self, map_i: np.ndarray) -> np.ndarray:
        """Crop the map to `hw_bd` and mask outside-rectangle area as occupied."""

        # cropped map and vertices
        map_cropped = copy.copy(map_i[self.hw_bd["h_min"]:self.hw_bd["h_max"], self.hw_bd["w_min"]:self.hw_bd["w_max"]])
        cropped_vertices = np.zeros_like(self.vertices)
        for i, v in enumerate(self.vertices):
            cropped_vertices[i] = np.array([v[0] - self.hw_bd["w_min"], v[1] - self.hw_bd["h_min"]])

        # make cells outside the rectangle as occupied
        mask = np.ones_like(map_cropped) * OCCUPIED_COST
        cv.fillPoly(mask, [cropped_vertices], FREE_COST)
        map_cropped = cv.bitwise_and(map_cropped, mask)

        # ensure binary
        map_cropped = map_cropped.astype(np.uint8)
        _, map_cropped = cv.threshold(map_cropped, 1, 255, cv.THRESH_BINARY)
        return map_cropped

    def make_frame(self, map_i: np.ndarray, local_frame: bool = True, value: int = 0) -> np.ndarray:
        """Draw the rectangle frame on a map with the given value/color."""
        if local_frame:
            hw_bd = self.hw_bd_local
        else:
            hw_bd = self.hw_bd

        # thickness and color
        thickness = 6
        if not value:
            value = [0, 0, 250]

        map_with_rect = cv.rectangle(map_i.copy(),
                                     (hw_bd["w_min"], hw_bd["h_min"]),  # Top-left corner
                                     (hw_bd["w_max"], hw_bd["h_max"]),  # Bottom-right corner
                                     value, thickness)
        # map_i[self.hw_bd[''], 0] = value
        # map_i[:, -1] = value
        # map_i[0, :] = value
        # map_i[-1, :] = value
        return map_with_rect

    def apply_hw_limits(self, hw_limits) -> dict:
        """Clamp rectangle bounds to fit within the original map dimensions.

        Returns:
        - dict: Updated bounds with keys h_min, w_min, h_max, w_max.
        """
        updated_hw = copy.copy(hw_limits)
        w_min = max(0, updated_hw["w_min"])
        h_min = max(0, updated_hw["h_min"])
        w_max = min(updated_hw["w_max"], self.orig_map_data.updated_w-1)
        h_max = min(updated_hw["h_max"], self.orig_map_data.updated_h-1)
        updated_hw = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        return updated_hw

    # ==========================================================================

    def dilate_erode_map(self, map_i: np.ndarray, radius_m: float, erode: bool = False) -> np.ndarray:
        """Morphologically dilate or erode the map by a metric radius.

        Args:
        - map_i: Input binary map.
        - radius_m: Metric radius to apply (converted using map resolution).
        - erode: When True applies erosion; otherwise applies dilation.

        Returns:
        - np.ndarray: Binary result after morphology.
        """
        radius_px = int(radius_m / self.orig_map_data.up_resolution)
        kernel_size = 2*radius_px + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if erode:
            map_dilated = cv.erode(map_i, kernel, iterations=1)
        else:
            map_dilated = cv.dilate(map_i, kernel, iterations=1)
        map_dilated = np.where(map_dilated > 0, 255, 0).astype(np.uint8)
        return map_dilated

    def display_contour(self, c) -> None:
        """Display all contours from findContours, each contour with a different color. """
        # Create an empty image to draw contours
        contour_img = np.zeros_like(self.map_c_orig)
        contour_img_rgb = cv.cvtColor(contour_img, cv.COLOR_GRAY2BGR)

        # Generate random colors for each contour
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(c))]

        # Draw each contour with a different color
        for i, contour in enumerate(c):
            cv.drawContours(contour_img_rgb, [contour], -1, colors[i], 2)

        # Display the image with contours
        cv.imshow("Contours", contour_img_rgb)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def add_rectangle(self, map_i: np.ndarray, vertices: np.ndarray, color=None, thickness: int = 6) -> np.ndarray:
        """ Add rectangle to the map image. """
        if not color:
            color = [0, 0, 250]
        map_with_rect = cv.polylines(map_i, [vertices], isClosed=True, color=color, thickness=thickness)
        return map_with_rect

    def create_final_vis_map(self) -> np.ndarray:
        """
        Create final visualization map.
        Display:
            - occupied area in ws map - dilated (red)
            - uncleaned area in cleaning map - dilated (green)
            - inaccessible (but free) (yellow)
            - final free space (white)
        """

        # occupied area in ws map - dilated
        infeasible_area = cv.subtract(self.map_c_orig, self.map_4_c_dilated_ws)
        infeasible_area_inds = infeasible_area == FREE_COST

        # uncleaned area in cleaning map - dilated
        uncleaned_area = cv.subtract(self.map_c_orig, self.map_3_c_dilated_cleaning)
        uncleaned_area_inds = uncleaned_area == FREE_COST

        # inaccessible (but free)
        inaccessible_area = cv.subtract(self.map_3_c_dilated_cleaning, self.map_11_c_ws_sf_f_cleaning)
        inaccessible_area_inds = inaccessible_area == FREE_COST

        # final free space
        free_selected_inds = self.map_3_c_dilated_cleaning == FREE_COST

        # create visualization map
        map_viz = copy.copy(self.binary_map_0[self.hw_bd["h_min"]:self.hw_bd["h_max"], self.hw_bd["w_min"]:self.hw_bd["w_max"]])
        map_viz = cv.cvtColor(map_viz, cv.COLOR_GRAY2BGR)
        # map_viz[occ_cropped_inds] = [0, 0, 0]
        map_viz[free_selected_inds] = [250, 250, 250]
        map_viz[infeasible_area_inds] = [250, 0, 0]
        map_viz[uncleaned_area_inds] = [0, 200, 0]
        map_viz[inaccessible_area_inds] = [250, 250, 0]

        # crop size
        h_c, w_c = self.map_c_orig.shape

        # plot on main map
        if self.plot_main:
            x, y = self.hw_bd["h_min"], self.hw_bd["w_min"]
            map_main_viz = cv.cvtColor(self.binary_map_0, cv.COLOR_GRAY2BGR)
            map_main_viz[x:x+h_c, y:y+w_c] = map_viz

            # # plot final rectangle
            # if self.final_vertices is not None:
            #     # Perimeter rectangle on free space
            #     map_main_viz = self.add_rectangle(map_main_viz, self.final_vertices, color=[0, 0, 200], thickness=4)
        else:
            map_main_viz = map_viz

        # vis robot
        viz_robot = False
        if viz_robot:
            sh_x = 130
            sh_y = 25
            alpha = 0.5
            shape_main_viz = self.binary_map_0.shape
            pose = (shape_main_viz[1]//2+sh_x - sh_y, shape_main_viz[0]//2 + sh_y)
            radius_px_1 = int(np.ceil(self.scfg.ws_r_dist / self.orig_map_data.up_resolution))
            radius_px_2 = int(np.ceil(self.scfg.robot_radius / self.orig_map_data.up_resolution))
            radius_px_3 = int(np.ceil(self.scfg.cleaning_r / self.orig_map_data.up_resolution))
            overlay = map_main_viz.copy()
            cv.circle(overlay, pose, radius_px_1, [0, 250, 0], -1)
            cv.circle(overlay, pose, radius_px_2, [0, 0, 250], -1)
            cv.circle(overlay, pose, radius_px_3, [250, 100, 0], -1)
            cv.circle(overlay, pose, 5, [250, 250, 250], -1)
            map_main_viz = cv.addWeighted(overlay, alpha, map_main_viz, 1 - alpha, 0)

        #
        # visualize_map_s(map_main_viz, "map_viz")
        # cv.imwrite(self.res_path + "/" + "map_main_viz.png", map_main_viz)

        return map_main_viz
