import copy
import cv2 as cv
import numpy as np
from scripts.zone import ZoneMapData
from scripts.base.map_data import MapData
from scripts.gui.visuals import Rectangle
from scripts.base.configs import SpaceConfigs
# from scripts.base.my_utils import visualize_map_s

FREE_COST = 255
OCCUPIED_COST = 0


class SpaceAnalyzer:
    map_data: MapData
    vertices: np.ndarray
    hw_bd: dict
    fs_selected_point: tuple[float, float]
    bb_c_ws_sf_f_cleaning: dict
    binary_map: np.ndarray
    cropped_mask: np.ndarray
    map_dilated_cleaning: np.ndarray
    map_ws_sf_f_orig: np.ndarray
    map_ws_sf_f_cleaning: np.ndarray
    map_c_orig: np.ndarray
    map_c_dilated_cleaning: np.ndarray
    map_c_ws_sf_f_orig: np.ndarray
    map_c_ws_sf_f_cleaning: np.ndarray
    __zone_map_data: ZoneMapData

    def __init__(self, map_data: MapData, rectangle: Rectangle):
        self.ns = "SpaceAnalyzer"
        print(f"[{self.ns}]: Analyzing space for creating free space [local map].")

        # configs
        self.scfg = SpaceConfigs()
        self.simple = True
        self.plot_main = True

        # inputs
        self.binary_map = map_data.up_bin_map     # binary map
        self.basic_map_data = map_data
        self.vertices = np.array(rectangle.vertices_t, dtype=np.int32)  # district vertices
        self.__zone_map_data = None

        # attributes
        self.hw_bd: dict[str:int] = None       # cropped hw limits
        self.is_fs_point_selected = False
        self.fs_selected_point = None
        self.bb_c_ws_sf_f_cleaning = None

        # maps
        self.map_c_orig = None
        self.cropped_mask = None
        self.map_dilated_cleaning = None
        self.map_ws_sf_f_orig = None
        self.map_ws_sf_f_cleaning = None
        self.map_c_dilated_cleaning = None
        self.map_c_ws_sf_f_orig = None
        self.map_c_ws_sf_f_cleaning = None
        self.map_viz = None

        # process
        self.get_vertices_data()

        # final
        print(f"[{self.ns}]: ----------------------------------------------- ")

    @property
    def zone_map_data(self) -> ZoneMapData:
        return copy.deepcopy(self.__zone_map_data)

    def update_by_rectangle(self, rectangle: Rectangle) -> None:
        self.vertices = np.array(rectangle.vertices_t, dtype=np.int32)
        self.get_vertices_data()
        self.map_c_orig = self.crop_map(self.binary_map)
        self.map_c_dilated_cleaning = self.crop_map(self.map_dilated_cleaning)
        self.map_c_ws_sf_f_orig = self.crop_map(self.map_ws_sf_f_orig)
        self.map_c_ws_sf_f_cleaning = self.crop_map(self.map_ws_sf_f_cleaning)

        # final map visualization
        self.map_viz = self.final_space_data()

        # create district object
        self.create_district()

    def create_local_map_ws_selection(self) -> None:

        # crop map and mask
        binary_map = self.binary_map
        self.map_c_orig = self.crop_map(binary_map)
        self.cropped_mask = np.ones_like(self.map_c_orig) * FREE_COST

        # detect workspace (based on robot size and distance conditions)
        if self.simple:
            # simple version - dilate obstacles
            map_dilated_ws = self.dilate_erode_map(map_i=binary_map, radius_m=self.scfg.ws_r_dist, erode=True)
            map_dilated_cleaning = self.dilate_erode_map(map_i=binary_map, radius_m=self.scfg.cleaning_dist, erode=True)
            map_c_dilated_ws = self.crop_map(map_dilated_ws)
            map_c_dilated_cleaning = self.crop_map(map_dilated_cleaning)
        else:
            # apply distance conditions using distance transform
            map_c_dilated_ws = self.distance_filter(map_full=binary_map)
            map_c_dilated_cleaning = self.distance_filter(map_full=binary_map)
        self.map_dilated_cleaning = map_dilated_cleaning
        self.map_c_dilated_cleaning = map_c_dilated_cleaning

        # select free space point
        self.select_free_space(map_c_dilated_ws)
        map_ws_sf = self.seperate_selected_free_space(map_dilated_ws)  # selected work space
        # map_c_ws_sf

        # remove islands
        map_ws_sf_f = self.remove_islands(map_ws_sf)
        # map_c_ws_sf_f
        # visualize_map_s(map_sws_rif, "map_sws_rif", False)

        # get cleaning space
        self.map_ws_sf_f_orig = self.dilate_erode_map(map_i=map_ws_sf_f, radius_m=self.scfg.ws_r_dist, erode=False)
        self.map_ws_sf_f_cleaning = self.dilate_erode_map(map_i=map_ws_sf_f, radius_m=self.scfg.effective_cleaning_r, erode=False)

        # maps
        self.map_c_ws_sf_f_orig = self.crop_map(self.map_ws_sf_f_orig)
        self.map_c_ws_sf_f_cleaning = self.crop_map(self.map_ws_sf_f_cleaning)
        bb_c_ws_sf_f_cleaning = self.get_freespace_bb(self.map_c_ws_sf_f_cleaning)
        self.bb_c_ws_sf_f_cleaning = bb_c_ws_sf_f_cleaning
        # visualize_map_s(map_sws_rif_cleaning, "map_sws_rif_cleaning")

        # final map visualization
        self.map_viz = self.final_space_data()

        # create district object
        self.create_district()

    def create_district(self) -> None:
        zone_map_data = ZoneMapData()
        zone_map_data.set_map_data(self.basic_map_data)
        zone_map_data.set_map_orig(self.map_c_orig)
        zone_map_data.set_map_sf_orig(self.map_c_ws_sf_f_orig)
        zone_map_data.set_map_cleaning(self.map_c_ws_sf_f_cleaning)
        zone_map_data.set_map_viz(self.map_viz)
        zone_map_data.set_bb_px(self.hw_bd)
        self.__zone_map_data = zone_map_data

    def get_freespace_bb(self, map_i: np.ndarray) -> dict:
        """get free selected bounding box
        """
        radius_px = int(np.ceil(self.scfg.cleaning_dist / self.basic_map_data.up_resolution))
        kernel_size = 2*radius_px + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        map_dilated = cv.dilate(map_i, kernel)
        map_dilated = np.where(map_dilated > 0, 255, 0).astype(np.uint8)
        bbd = self.find_bounding_box(map_dilated)
        bbd = self.apply_hw_limits(bbd)
        return bbd

    def remove_islands(self, map_i: np.ndarray) -> np.ndarray:
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

    def seperate_selected_free_space(self, c_map: np.ndarray) -> np.ndarray:
        """ selecting open space from candidates based on the clicked point.
        find all contours in the image and selects contour which contains selected point.
        each contour will be separate free space.
        Args:
            c_map (np.ndarray): cropped map.
        Returns:
            np.ndarray: selected_free_space.
        """
        print(f"[{self.ns}]: selecting the open space based on the selecting point.")

        # get contours
        contours, hierarchy = cv.findContours(c_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # pylint: disable=W0612
        if len(contours) == 0:
            print(f"[{self.ns}]: contours is empty")
            return c_map

        # # display contours
        # self.display_contour(contours)

        # find selected_contour
        print(f"[{self.ns}]: number of contours (free islands): {len(contours)}")
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

        #
        print(f"[{self.ns}]: free space is selected.")
        return map_fs

    def select_free_space(self, map_i: np.ndarray) -> None:
        print(f"[{self.ns}]: selecting the free space ...")
        free_map_disp = cv.addWeighted(self.map_c_orig, 0.4, map_i, 0.6, 0)
        win_name = "select free space"
        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(win_name, self.select_point)
        cv.imshow(win_name, free_map_disp)
        while not self.is_fs_point_selected:
            cv.waitKey(1)
        cv.destroyAllWindows()

    def distance_filter(self, map_full: np.ndarray) -> np.ndarray:
        return self.detect_free_spaces_dist_occ(map_full=map_full, dist_from_occ_m=self.scfg.ws_r_dist)

    def detect_free_spaces_dist_occ(self, map_full: np.ndarray, dist_from_occ_m: float = 0.0) -> np.ndarray:
        print(f"[{self.ns}]: detecting all free spaces conidering distance {dist_from_occ_m} from obstacles.")
        # current map
        if map_full is None:
            map_full = self.binary_map

        # Area for appling distance transform - expanding the cropped map - e -> expanded
        # Additional distance is necessary to check if any obstacle isn't just outside of the zone border
        dist_from_occ_px = int(np.ceil(dist_from_occ_m / self.basic_map_data.up_resolution))
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

        #
        # visualize_map(map_ws, "map_ws")
        return map_ws

    def get_vertices_data(self) -> dict:
        wh_max = np.max(self.vertices, axis=0)
        wh_min = np.min(self.vertices, axis=0)
        w_max, h_max = wh_max.tolist()
        w_min, h_min = wh_min.tolist()
        # corners
        hw_limits = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        hw_bd = self.apply_hw_limits(hw_limits)
        self.hw_bd = hw_bd

    def crop_map(self, map_i: np.ndarray) -> np.ndarray:
        # print(f"[{self.ns}]: cropping map based on the district rectangle.")

        # cropped map and vertices
        map_cropped = copy.copy(map_i[self.hw_bd["h_min"]:self.hw_bd["h_max"], self.hw_bd["w_min"]:self.hw_bd["w_max"]])
        cropped_vertices = np.zeros_like(self.vertices)
        for i, v in enumerate(self.vertices):
            cropped_vertices[i] = np.array([v[0] - self.hw_bd["w_min"], v[1] - self.hw_bd["h_min"]])

        # make cells outside the district as occupied
        mask = np.ones_like(map_cropped) * OCCUPIED_COST
        cv.fillPoly(mask, [cropped_vertices], FREE_COST)
        map_cropped = cv.bitwise_and(map_cropped, mask)

        # ensure binary
        map_cropped = map_cropped.astype(np.uint8)
        _, map_cropped = cv.threshold(map_cropped, 1, 255, cv.THRESH_BINARY)
        return map_cropped

    def make_frame(self, map_i: np.ndarray, value: int = 0):  # todo asap
        """ make frame of the map as value (default = 0) """
        map_i = self.add_rectangle_bd(map_i, self.hw_bd, color=value)
        # map_i[self.hw_bd[''], 0] = value
        # map_i[:, -1] = value
        # map_i[0, :] = value
        # map_i[-1, :] = value
        return map_i

    def apply_hw_limits(self, hw_limits) -> dict:
        updated_hw = copy.copy(hw_limits)
        w_min = max(0, updated_hw["w_min"])
        h_min = max(0, updated_hw["h_min"])
        w_max = min(updated_hw["w_max"], self.basic_map_data.updated_w-1)
        h_max = min(updated_hw["h_max"], self.basic_map_data.updated_h-1)
        updated_hw = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        return updated_hw

    def select_point(self, event, x, y, _, __) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            self.fs_selected_point = (x, y)
            self.is_fs_point_selected = True

    def dilate_erode_map(self, map_i: np.ndarray, radius_m: float, erode: False) -> tuple[np.ndarray, np.ndarray]:
        radius_px = int(np.ceil(radius_m / self.basic_map_data.up_resolution))
        kernel_size = 2*radius_px + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if erode:
            map_dilated = cv.erode(map_i, kernel)
        else:
            map_dilated = cv.dilate(map_i, kernel)
        map_dilated = np.where(map_dilated > 0, 255, 0).astype(np.uint8)
        # new_occupied_area = cv.subtract(map_i, map_dilated)
        # occ_new_added_inds = new_occupied_area == FREE_COST
        return map_dilated

    def display_contour(self, c) -> None:
        """
        Display all contours from findContours, each contour with a different color.

        Args:
            c (list): List of contours.
        """
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

    def find_bounding_box(self, map_i: np.ndarray) -> dict:
        free_cells = np.where(map_i == 255)
        if free_cells[0].size == 0:
            return None

        # Compute the bounding box
        h_min, h_max = free_cells[0].min(), free_cells[0].max()
        w_min, w_max = free_cells[1].min(), free_cells[1].max()
        hw = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        return hw

    def add_rectangle_bd(self, map_i: np.ndarray, hw_bd: dict, color=None):
        if not color:
            color = [0, 0, 250]
        map_with_rect = cv.rectangle(
            map_i.copy(),
            (hw_bd["w_min"], hw_bd["h_min"]),  # Top-left corner
            (hw_bd["w_max"], hw_bd["h_max"]),  # Bottom-right corner
            color,  # Color of the rectangle (gray)
            2  # Thickness of the rectangle
        )
        return map_with_rect

    def convert_map_to_world(self, p_m: tuple[int, int]) -> tuple[float, float]:  # todo todel
        """[x, y] [w, h] from map (pixels) to world (meters)"""
        p_w_0 = np.round(p_m[0] * self.basic_map_data.up_resolution + self.basic_map_data.origin_m[0], 4)
        p_w_1 = np.round(p_m[1] * self.basic_map_data.up_resolution + self.basic_map_data.origin_m[1], 4)
        return (p_w_0, p_w_1)

    def final_space_data(self) -> None:
        # #
        # occ_cropped_inds = self.map_c_orig == OCCUPIED_COST

        # new occupied area - dilated
        new_occupied_area = cv.subtract(self.map_c_orig, self.map_c_dilated_cleaning)
        occ_new_added_inds = new_occupied_area == FREE_COST

        # map free other - now occupied - no access
        map_free_other = cv.subtract(self.map_c_dilated_cleaning, self.map_c_ws_sf_f_cleaning)
        free_other_inds = map_free_other == FREE_COST

        # final free space
        free_selected_inds = self.map_c_dilated_cleaning == FREE_COST

        #
        map_viz = np.zeros_like(self.map_c_orig) + 0
        map_viz = cv.cvtColor(map_viz, cv.COLOR_GRAY2BGR)
        # map_viz[occ_cropped_inds] = [0, 0, 0]
        map_viz[free_selected_inds] = [250, 250, 250]
        map_viz[occ_new_added_inds] = [250, 0, 0]
        map_viz[free_other_inds] = [250, 250, 0]

        #
        h_c, w_c = self.map_c_orig.shape
        if self.bb_c_ws_sf_f_cleaning:
            map_viz = self.add_rectangle_bd(map_viz, self.bb_c_ws_sf_f_cleaning, color=[0, 0, 250])
            hw_bd_crop = {"h_min": 0, "w_min": 0, "h_max": h_c, "w_max": w_c}
            map_viz = self.add_rectangle_bd(map_viz, hw_bd_crop, color=[0, 250, 0])

        #
        if self.plot_main:
            x, y = self.hw_bd["h_min"], self.hw_bd["w_min"]
            map_main_viz = cv.cvtColor(self.binary_map, cv.COLOR_GRAY2BGR)
            map_main_viz[x:x+h_c, y:y+w_c] = map_viz
        else:
            map_main_viz = map_viz

        # vis robot
        sh = 17
        alpha = 0.5
        shape_main_viz = self.binary_map.shape
        pose = (shape_main_viz[1]//2 - sh, shape_main_viz[0]//2 + sh)
        radius_px_1 = int(np.ceil(self.scfg.ws_r_dist / self.basic_map_data.up_resolution))
        radius_px_2 = int(np.ceil(self.scfg.effective_cleaning_r / self.basic_map_data.up_resolution))
        overlay = map_main_viz.copy()
        cv.circle(overlay, pose, radius_px_1, [0, 250, 0], -1)
        cv.circle(overlay, pose, radius_px_2, [0, 0, 250], -1)
        map_main_viz = cv.addWeighted(overlay, alpha, map_main_viz, 1 - alpha, 0)

        #
        # visualize_map_s(map_main_viz, "map_viz")
        # cv.imwrite(self.res_path + "/" + "map_main_viz.png", map_main_viz)

        return map_main_viz
