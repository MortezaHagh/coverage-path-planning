import os
import copy
from typing import Optional
import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter
from scripts.map_data import MapData
from scripts.gui.visuals import Rectangle
from scripts.my_utils import visualize_map
from scripts.configs import BConfigs, SpaceConfigs

FREE_COST = 255
OCCUPIED_COST = 0


class SpaceAnalyzer:
    map_data: MapData
    binary_map: np.ndarray
    vertices: np.ndarray
    # district_rect: Rectangle

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

        # inputs
        self.binary_map = map_data.up_bin_map     # binary map
        self.map_data = map_data
        self.district_rect = district_rect
        self.vertices = np.array(district_rect.vertices_t, dtype=np.int32)  # district vertices

        # attributes
        self.c_hw_l: dict[str:int] = None       # cropped hw limits
        self.free_space_selected = False
        self.free_space_point = [0, 0]

        #
        self.map_cropped: np.ndarray = None
        self.cropped_mask: np.ndarray = None

        #
        self.select_space()

        # #
        # zone = MapInfo(free_space_map, available_map, zone_coordinates, mapPgm.shape, mapOrigin)

        # final
        print(f"[{self.ns}]: ----------------------------------------------- ")

    def select_space(self):

        # crop map
        map_cropped = self.crop_map(self.binary_map)
        self.map_cropped = map_cropped
        self.cropped_mask = np.ones_like(map_cropped) * FREE_COST

        # self.detect_narrow(map_cropped)
        # return

        if self.simple:
            # simple version
            map_dilated, _ = self.dilate_map(map_i=self.binary_map, radius_m=self.scfg.free_space_obst_dist)
            map_free_dis_f = self.crop_map(map_dilated)
        else:
            # Find Free Space
            map_free_dis_f = self.distance_filter()

        # select free space
        self.select_free_space(map_free_dis_f)
        map_free_selected = self.seperate_selected_free_space(map_free_dis_f)
        visualize_map(map_free_selected, "map_free_selected")

        # remove islands
        map_island_f = self.remove_islands(map_free_selected)

        #
        self.final_space_data(map_free_dis_f, map_island_f)

    def remove_islands(self, map_i: np.ndarray) -> np.ndarray:
        contours, hierarchy = cv.findContours(map_i, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return map_i

        # remove area of a contour with parent contour
        map_island_f = np.ones_like(map_i) * FREE_COST
        for i, element in enumerate(hierarchy[0]):

            if (element[3] != -1):
                cv.drawContours(map_island_f, [contours[i]], 0, OCCUPIED_COST, cv.FILLED)
        map_island_f = cv.bitwise_and(map_i, map_island_f)
        return map_island_f

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
            if cv.pointPolygonTest(contour, self.free_space_point, True) >= 0:
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
        map_free_selected = cv.bitwise_and(c_map, selected_free_space)

        #
        print(f"[{self.ns}]: free space is selected.")
        return map_free_selected

    def select_free_space(self, map_i: np.ndarray) -> None:
        print(f"[{self.ns}]: selecting the free space ...")
        free_map_disp = cv.addWeighted(self.map_cropped, 0.4, map_i, 0.6, 0)
        win_name = "select free space"
        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(win_name, self.select_point)
        cv.imshow(win_name, free_map_disp)
        while not self.free_space_selected:
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
        e_hw_l = copy.copy(self.c_hw_l)
        e_hw_l["w_min"] -= e_min_obst_dist
        e_hw_l["h_min"] -= e_min_obst_dist
        e_hw_l["w_max"] += e_min_obst_dist
        e_hw_l["h_max"] += e_min_obst_dist
        e_hw_l = self.apply_hw_limits(e_hw_l)
        d_h_min = self.c_hw_l["h_min"] - e_hw_l["h_min"]
        d_w_min = self.c_hw_l["w_min"] - e_hw_l["w_min"]

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
        map_free_dis_f = copy.copy(self.cropped_mask)
        map_free_dis_f[dist_trans < norm_dist_from_occ_px] = OCCUPIED_COST

        #
        # visualize_map(map_free_dis_f, "map_free_dis_f")
        return map_free_dis_f

    def crop_map(self, map_i: np.ndarray) -> np.ndarray:
        print(f"[{self.ns}]: cropping map based on the district rectangle.")

        # bounds
        wh_max = np.max(self.vertices, axis=0)
        wh_min = np.min(self.vertices, axis=0)
        w_max, h_max = wh_max.tolist()
        w_min, h_min = wh_min.tolist()

        # corners
        hw_limits = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        c_hw_l = self.apply_hw_limits(hw_limits)
        self.c_hw_l = c_hw_l
        # self.cleaningZoneBBox = [h_min, h_max, w_min, w_max]

        # cropped map and vertices
        map_cropped = copy.copy(map_i[c_hw_l["h_min"]:c_hw_l["h_max"], c_hw_l["w_min"]:c_hw_l["w_max"]])
        cropped_vertices = np.zeros_like(self.vertices)
        for i, v in enumerate(self.vertices):
            cropped_vertices[i] = np.array([v[0] - w_min, v[1] - h_min])

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
            self.free_space_point = [x, y]
            self.free_space_selected = True

    def dilate_map(self, map_i: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        radius_px = int(np.ceil(radius_m / self.map_data.up_resolution))
        kernel_size = 2*radius_px + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        map_dilated = cv.erode(map_i, kernel)
        new_occupied_area = cv.subtract(map_i, map_dilated)
        occ_new_added_inds = new_occupied_area == FREE_COST
        return map_dilated, occ_new_added_inds

    def final_space_data(self, map_free_dis_f: np.ndarray = None, map_free_selected: np.ndarray = None):
        print(f"[{self.ns}]: Visualizing workspace.")
        #
        occ_cropped_inds = self.map_cropped == OCCUPIED_COST

        #
        new_occupied_area = cv.subtract(self.map_cropped, map_free_dis_f)
        occ_new_added_inds = new_occupied_area == FREE_COST

        #
        map_free_other = cv.subtract(map_free_dis_f, map_free_selected)
        free_other_inds = map_free_other == FREE_COST

        #
        free_selected_inds = map_free_dis_f == FREE_COST

        #
        viz = np.zeros_like(self.cropped_mask) + 0
        viz_rgb = cv.cvtColor(viz, cv.COLOR_GRAY2BGR)
        # viz_rgb[occ_cropped_inds] = [0, 0, 0]
        viz_rgb[free_selected_inds] = [250, 250, 250]
        viz_rgb[occ_new_added_inds] = [250, 0, 0]
        viz_rgb[free_other_inds] = [250, 250, 0]

        #
        visualize_map(viz_rgb, "viz_rgb")

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
