import os
import copy
import cv2 as cv
import numpy as np
from typing import Optional
from scripts.map_data import MapData
from scripts.gui.visuals import Rectangle
from scripts.my_utils import visualize_map
from scripts.configs import BConfigs, SpaceConfigs


class SpaceAnalyzer:
    map_data: MapData
    binary_map: np.ndarray
    vertices: np.ndarray
    # district_rect: Rectangle

    def __init__(self, map_data: MapData, bconfigs: BConfigs, district_rect: Rectangle):
        self.ns = "SpaceAnalyzer"

        # analyzed space path
        self.res_path = os.path.join(bconfigs.map_path, bconfigs.district_folder)
        os.makedirs(self.res_path, exist_ok=True)

        # configs
        self.bcfg = bconfigs
        self.scfg = SpaceConfigs()

        # inputs
        self.binary_map = map_data.up_bin_map     # binary map
        self.map_data = map_data
        self.district_rect = district_rect
        self.vertices = np.array(district_rect.vertices_t, dtype=np.int32)  # district vertices

        # attributes
        self.c_hw_l: dict[str:int] = None

        #
        self.select_space()

        # #
        # zone = MapInfo(free_space_map, available_map, zone_coordinates, mapPgm.shape, mapOrigin)

    def select_space(self):
        # print(f"[{self.ns}]: vertices:", self.vertices)

        # crop map
        cropped_map, out_mask = self.crop_map()

        # Find Free Space
        free_map = self.detect_free_space(out_mask)
        free_map_disp = cv.addWeighted(cropped_map, 0.4, free_map, 0.6, 0)
        visualize_map(free_map_disp, "free_map_disp")

    def detect_free_space(self, out_mask: np.ndarray, map_main: Optional[np.ndarray] = None):
        # current map
        if map_main is None:
            map_main = self.binary_map

        # todo: we could apply on the main_map without other calculations

        # Area for appling distance transform - expanding the cropped map
        # Additional distance is necessary to check if any obstacle isn't just outside of the zone border
        # e -> expanded
        wf_min_obst_dist_px = int(np.ceil(self.scfg.wf_min_obst_dist / self.map_data.up_resolution))
        c_min_obst_dist = 2
        e_min_obst_dist = wf_min_obst_dist_px * c_min_obst_dist
        e_hw_l = copy.copy(self.c_hw_l)
        e_hw_l["w_min"] -= e_min_obst_dist
        e_hw_l["h_min"] -= e_min_obst_dist
        e_hw_l["w_max"] += e_min_obst_dist
        e_hw_l["h_max"] += e_min_obst_dist
        e_hw_l = self.apply_hw_limits(e_hw_l)
        d_h_min = self.c_hw_l["h_min"] - e_hw_l["h_min"]
        d_w_min = self.c_hw_l["w_min"] - e_hw_l["w_min"]

        # expanded cropped map for distance calculation
        e_c_map = map_main[e_hw_l["h_min"]:e_hw_l["h_max"], e_hw_l["w_min"]:e_hw_l["w_max"]]

        # computing distance transform for local map (cropped map)
        dist_trans = cv.distanceTransform(e_c_map, distanceType=cv.DIST_L2, maskSize=5, dstType=cv.CV_8U)
        max_dist_px = np.amax(dist_trans)
        dist_trans = cv.normalize(dist_trans, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # free space distance limit
        free_space_obst_dist_lim_px = np.ceil(wf_min_obst_dist_px * 255 / max_dist_px)
        # print("free_space_obst_dist_lim_px", free_space_obst_dist_lim_px)

        # eliminate expanded part from dist_trans
        n_h, n_w = out_mask.shape
        dist_trans = dist_trans[d_h_min:d_h_min+n_h, d_w_min:d_w_min+n_w]

        # free map
        free_map = copy.copy(out_mask)
        free_map[dist_trans < free_space_obst_dist_lim_px] = 0
        # visualize_map(free_map, "free_map")

        # # Get cells with distance equal to free_space_obst_dist_lim_px and make them red
        # dist_trans_rgb = cv.cvtColor(dist_trans, cv.COLOR_GRAY2BGR)
        # red_cells = dist_trans == free_space_obst_dist_lim_px
        # dist_trans_rgb[red_cells] = [0, 0, 255]
        # visualize_map(dist_trans_rgb, "dist_trans_rgb_red")
        return free_map

    def crop_map(self):
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
        cropped_map = copy.copy(self.binary_map[c_hw_l["h_min"]:c_hw_l["h_max"], c_hw_l["w_min"]:c_hw_l["w_max"]])
        cropped_vertices = np.zeros_like(self.vertices)
        for i, v in enumerate(self.vertices):
            cropped_vertices[i] = np.array([v[0] - w_min, v[1] - h_min])
        # visualize_map(cropped_map)

        # outside mask
        out_mask = np.ones_like(cropped_map) * 255  # 255: white pixels

        # # outside mask
        # out_mask = np.zeros_like(cropped_map)
        # # fill mask with white pixels
        # cv.fillConvexPoly(out_mask, cropped_vertices, 255)
        # # replace pixel outside of the zone with black pixels
        # for hp in range(0, h_max - h_min):
        #     for wp in range(0, w_max - w_min):
        #         if out_mask[hp, wp] == 0:
        #             cropped_map[hp, wp] = 0
        #             # never happens!

        # # visualize
        # visualize_map(cropped_map, "cropped_map")
        # visualize_map(out_mask)

        return cropped_map, out_mask

    def apply_hw_limits(self, hw_limits):
        updated_hw = copy.copy(hw_limits)
        w_min = max(0, updated_hw["w_min"])
        h_min = max(0, updated_hw["h_min"])
        w_max = min(updated_hw["w_max"], self.map_data.updated_w-1)
        h_max = min(updated_hw["h_max"], self.map_data.updated_h-1)
        updated_hw = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        return updated_hw
