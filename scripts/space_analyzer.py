import os
import copy
from typing import Optional
import cv2 as cv
import numpy as np
from scripts.map_data import MapData
from scripts.gui.visuals import Rectangle
from scripts.configs import BConfigs, SpaceConfigs
from scripts.my_utils import visualize_map_s  # visualize_map

FREE_COST = 255
OCCUPIED_COST = 0


class SpaceAnalyzer:
    map_data: MapData
    vertices: np.ndarray
    hw_bd: dict
    fs_selected_point: tuple[float, float]
    binary_map: np.ndarray
    map_cropped: np.ndarray
    cropped_mask: np.ndarray

    def __init__(self, map_data: MapData, bconfigs: BConfigs, district_rect: Rectangle):
        self.ns = "SpaceAnalyzer"
        print(f"[{self.ns}]: Analyzing space for creating free space.")

        # analyzed space path
        self.res_path = os.path.join(bconfigs.map_path, bconfigs.district_folder)
        os.makedirs(self.res_path, exist_ok=True)

        # configs
        self.bcfg = bconfigs
        self.scfg = SpaceConfigs()
        self.simple = True
        self.plot_main = True

        # inputs
        self.binary_map = map_data.up_bin_map     # binary map
        self.map_data = map_data
        self.vertices = np.array(district_rect.vertices_t, dtype=np.int32)  # district vertices
        # self.district_rect = district_rect

        # attributes
        self.hw_bd: dict[str:int] = None       # cropped hw limits
        self.is_fs_point_selected = False
        self.fs_selected_point = None

        #
        self.map_cropped: np.ndarray = None
        self.cropped_mask: np.ndarray = None

        # ========================================
        self.select_space()

        # #
        # zone = MapInfo(free_space_map, available_map, zone_coordinates, mapPgm.shape, mapOrigin)

        # final
        print(f"[{self.ns}]: ----------------------------------------------- ")

    def select_space(self) -> None:

        # crop map and mask
        map_cropped = self.crop_map(self.binary_map)
        self.map_cropped = map_cropped
        self.cropped_mask = np.ones_like(map_cropped) * FREE_COST

        # detect workspace (based on robot size and distance conditions)
        if self.simple:
            # simple version - dilate obstacles
            map_dilated, _ = self.dilate_map(map_i=self.binary_map, radius_m=self.scfg.free_space_obst_dist)
            map_ws = self.crop_map(map_dilated)
        else:
            # apply distance conditions using distance transform
            map_ws = self.distance_filter()

        # select free space point
        self.select_free_space(map_ws)
        map_wfs = self.seperate_selected_free_space(map_ws)

        # remove islands
        map_fs_rif = self.remove_islands(map_wfs)
        visualize_map_s(map_fs_rif, "map_fs_rif")

        # vizualize processed space
        fs_bb = self.get_fs_bb(map_fs_rif)
        self.final_space_data(map_ws, map_fs_rif, fs_bb)

        # save selected free space
        cv.imwrite(self.res_path + "/" + "freeSpace.png", map_fs_rif)

    def get_fs_bb(self, map_i: np.ndarray) -> dict:
        """get free selected bounding box
        """
        radius_px = int(np.ceil(self.scfg.free_space_obst_dist / self.map_data.up_resolution))
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
        free_map_disp = cv.addWeighted(self.map_cropped, 0.4, map_i, 0.6, 0)
        win_name = "select free space"
        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(win_name, self.select_point)
        cv.imshow(win_name, free_map_disp)
        while not self.is_fs_point_selected:
            cv.waitKey(1)
        cv.destroyAllWindows()

    def distance_filter(self, map_full: Optional[np.ndarray] = None) -> np.ndarray:
        return self.detect_free_spaces_dist_occ(map_full=map_full, dist_from_occ_m=self.scfg.free_space_obst_dist)

    def detect_free_spaces_dist_occ(self, map_full: Optional[np.ndarray] = None, dist_from_occ_m: float = 0.0) -> np.ndarray:
        print(f"[{self.ns}]: detecting all free spaces conidering distance {dist_from_occ_m} from obstacles.")
        # current map
        if map_full is None:
            map_full = self.binary_map

        # Area for appling distance transform - expanding the cropped map - e -> expanded
        # Additional distance is necessary to check if any obstacle isn't just outside of the zone border
        dist_from_occ_px = int(np.ceil(dist_from_occ_m / self.map_data.up_resolution))
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

    def crop_map(self, map_i: np.ndarray) -> np.ndarray:
        print(f"[{self.ns}]: cropping map based on the district rectangle.")

        # bounds
        wh_max = np.max(self.vertices, axis=0)
        wh_min = np.min(self.vertices, axis=0)
        w_max, h_max = wh_max.tolist()
        w_min, h_min = wh_min.tolist()

        # corners
        hw_limits = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        hw_bd = self.apply_hw_limits(hw_limits)
        self.hw_bd = hw_bd
        # self.cleaningZoneBBox = [h_min, h_max, w_min, w_max]

        # cropped map and vertices
        map_cropped = copy.copy(map_i[hw_bd["h_min"]:hw_bd["h_max"], hw_bd["w_min"]:hw_bd["w_max"]])
        cropped_vertices = np.zeros_like(self.vertices)
        for i, v in enumerate(self.vertices):
            cropped_vertices[i] = np.array([v[0] - w_min, v[1] - h_min])

        # ensure binary
        map_cropped = map_cropped.astype(np.uint8)
        _, map_cropped = cv.threshold(map_cropped, 1, 255, cv.THRESH_BINARY)
        return map_cropped

    def apply_hw_limits(self, hw_limits) -> dict:
        updated_hw = copy.copy(hw_limits)
        w_min = max(0, updated_hw["w_min"])
        h_min = max(0, updated_hw["h_min"])
        w_max = min(updated_hw["w_max"], self.map_data.updated_w-1)
        h_max = min(updated_hw["h_max"], self.map_data.updated_h-1)
        updated_hw = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        return updated_hw

    def select_point(self, event, x, y, _, __) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            self.fs_selected_point = (x, y)
            self.is_fs_point_selected = True

    def dilate_map(self, map_i: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        radius_px = int(np.ceil(radius_m / self.map_data.up_resolution))
        kernel_size = 2*radius_px + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        map_dilated = cv.erode(map_i, kernel)
        map_dilated = np.where(map_dilated > 0, 255, 0).astype(np.uint8)
        new_occupied_area = cv.subtract(map_i, map_dilated)
        occ_new_added_inds = new_occupied_area == FREE_COST
        return map_dilated, occ_new_added_inds

    def display_contour(self, c) -> None:
        """
        Display all contours from findContours, each contour with a different color.

        Args:
            c (list): List of contours.
        """
        # Create an empty image to draw contours
        contour_img = np.zeros_like(self.map_cropped)
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
        # print("hw_bd", hw_bd)
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

    def final_space_data(self, map_ws: np.ndarray = None, map_fs: np.ndarray = None, fs_bb: dict = None) -> None:
        print(f"[{self.ns}]: Visualizing workspace.")
        # #
        # occ_cropped_inds = self.map_cropped == OCCUPIED_COST

        #
        new_occupied_area = cv.subtract(self.map_cropped, map_ws)
        occ_new_added_inds = new_occupied_area == FREE_COST

        #
        map_free_other = cv.subtract(map_ws, map_fs)
        free_other_inds = map_free_other == FREE_COST

        #
        free_selected_inds = map_ws == FREE_COST

        #
        viz = np.zeros_like(self.cropped_mask) + 0
        sel_rgb = cv.cvtColor(viz, cv.COLOR_GRAY2BGR)
        # sel_rgb[occ_cropped_inds] = [0, 0, 0]
        sel_rgb[free_selected_inds] = [250, 250, 250]
        sel_rgb[occ_new_added_inds] = [250, 0, 0]
        sel_rgb[free_other_inds] = [250, 250, 0]

        #
        if fs_bb:
            sel_rgb = self.add_rectangle_bd(sel_rgb, fs_bb, color=[0, 0, 250])

        #
        if self.plot_main:
            h, w = self.map_cropped.shape
            x, y = self.hw_bd["h_min"], self.hw_bd["w_min"]
            map_main_rgb = cv.cvtColor(self.binary_map, cv.COLOR_GRAY2BGR)
            map_main_rgb[x:x+h, y:y+w] = sel_rgb
        else:
            map_main_rgb = sel_rgb

        #
        visualize_map_s(map_main_rgb, "viz_rgb")
