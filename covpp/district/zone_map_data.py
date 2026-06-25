"""ZoneMapData container for per-zone maps and metadata.

Holds cropped maps, rectangles, pixel bounds, and result paths with values in
pixels by default and global map coordinates unless noted otherwise.

Steps:
- Initialize identifiers and folders via :meth:`set_result_path`
- Populate maps and rectangles from SpaceAnalyzer
- Save maps and JSON payload via :meth:`save_data`

Result:
- Ready-to-use inputs for segmentation, coverage planning, and evaluation.
"""
import os
import json
import copy
import cv2 as cv
import numpy as np
from covpp.gui.rectangle import Rectangle
from covpp.base.map_data import MapData
from covpp.base.configs import PipeLineConfig
from covpp.base.base_utils import visualize_map_s


def convert_to_native(data: dict) -> dict:
    """ Convert numpy types to native python types for json serialization. """
    return {k: int(v) if isinstance(v, (np.integer, int))
            else float(round(v, 2)) if isinstance(v, (np.floating, float))
            else v
            for k, v in data.items()}


class ZoneMapData:
    """Storage for zone-level map artifacts and metadata (global frame).

    Args:
    - ind: Zone index identifier.

    Attributes:
    - Paths: ``zone_dir``, ``maps_dir``, ``result_file``, ``method_results_dir``, ``segmentation_dir``
    - Geometry: ``orig_rectangle``, ``rectangle``, ``final_rectangle``, ``rect_data``, ``final_rect_data``
    - Pixels/bounds: ``vertices``, ``final_vertices``, ``hw_bd``, ``hw``, ``global_xy``
    - Maps: ``map_orig``, ``map_ws``, ``map_sf_orig``, ``map_cleaning``, ``map_planning``, ``map_segmentation``, ``map_ws_segmented``
    """

    def __init__(self, ind: int = 0):
        self.ind: int = ind
        self.name: str = f"Zone_{self.ind}"
        self.zone_dir: str = None                   # path to zone directory ./maps/<map_name>/District/Zone_i
        self.maps_dir: str = None                   # path to zone maps directory ./maps/<map_name>/District/Zone_i/maps
        self.result_file: str = None                # path to zone result file ./maps/<map_name>/District/Zone_i/<method>/result.json
        self.method_results_dir: str = None         # path to zone result directory ./maps/<map_name>/District/Zone_i/<method>
        self.segmentation_dir: str = None           # path to zone segmentation directory ./maps/<map_name>/District/Zone_i/segmentation
        self.segmentation_method_dir: str = None           # path to zone segmentation directory ./maps/<map_name>/District/Zone_i/segmentation
        #
        self.resolution: float = None
        self.orig_map_data: MapData = None
        self.orig_rectangle: Rectangle = None
        self.rectangle: Rectangle = None                # current main zone or district rectangle
        self.final_rectangle: Rectangle = None          # final rectangle (free space bounding box)
        self.rect_data: dict = None                     # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self._final_rect_data: dict = None              # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self.vertices: list[tuple[int, int]] = None     # vertices: [(x, y), ...]
        self.final_vertices: np.ndarray = None          # vertices: [(x, y), ...]
        self.hw_bd: dict = None                         # map hw bound, dict keys: h_min, w_min, h_max, w_max
        self.hw: tuple[int, int] = None                 # height and width of the zone map
        self.global_xy: np.ndarray = None               # global map offset (x_offset, y_offset)
        # all maps are uint8, binary (OCCUPIED, FREE)
        # maps [fixed, for visualization, planning, and evaluation]
        self.map_orig: np.ndarray = None                # cropped original map based on rectangle
        self.map_ws: np.ndarray = None                  # cropped work space selected free space filtered (map_7_c_ws_sf_f)
        self.map_sf_orig: np.ndarray = None             # cropped work space selected free space filtered original
        self.map_cleaning: np.ndarray = None            # c work space selected free space filtered cleaning (map_11_c_ws_sf_f_cleaning)
        self.map_cleaned: np.ndarray = None             # map with cleaned area added
        # maps[updated during planning]
        self.map_planning: np.ndarray = None            # map visualization for planning (map_viz)
        self.map_segmentation: np.ndarray = None        # map segmentation visualization
        self.map_ws_segmented: np.ndarray = None        # map with segment lines added

    @property
    def final_rect_data(self) -> dict:
        """Return a shallow copy of the final rectangle data dict."""
        return copy.copy(self._final_rect_data)

    def set_name(self, name: str):
        self.name = name
        self.set_result_path()

    def set_ind(self, ind: int):
        self.ind = ind
        self.name = f"Zone_{self.ind}"
        self.set_result_path()

    def set_result_path(self):
        """Compute and create result folders based on map and zone name.

        Steps:
        - For district vs zone, build directory structure
        - Create segmentation and method result folders
        - Create maps subdirectory
        """

        if self.name == 'District':
            self.zone_dir = self.orig_map_data.mcfg.district_dir
            self.result_file = self.zone_dir + f"/result_{PipeLineConfig.method}.json"
        else:
            self.zone_dir = os.path.join(self.orig_map_data.mcfg.district_dir, self.name)
            self.result_file = self.zone_dir + f"/{PipeLineConfig.method}/result.json"
            self.segmentation_dir = os.path.join(self.zone_dir, "segmentation")
            os.makedirs(self.segmentation_dir, exist_ok=True)
            self.segmentation_method_dir = os.path.join(self.segmentation_dir, PipeLineConfig.segmentation_method)
            os.makedirs(self.segmentation_method_dir, exist_ok=True)
            self.method_results_dir = os.path.join(self.zone_dir, PipeLineConfig.method)
            os.makedirs(self.method_results_dir, exist_ok=True)

        # create zone maps directory
        os.makedirs(self.zone_dir, exist_ok=True)
        # self.maps_dir = os.path.join(self.zone_dir, "maps")
        # os.makedirs(self.maps_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Setters (called from space analyzer)
    # --------------------------------------------------------------------------

    def set_orig_map_data(self, map_data: MapData):
        self.orig_map_data = map_data
        self.resolution = map_data.up_resolution
        self.set_result_path()

    def set_orig_rectangle(self, rectangle: Rectangle):
        self.orig_rectangle = rectangle

    def set_rectangle(self, rectangle: Rectangle):
        self.rectangle = rectangle

    def set_rect_data(self, rect_data_t: dict, vertices: list[tuple[int, int]]):
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        # vertices: [(x, y), ...]
        self.rect_data = rect_data_t
        self.vertices = vertices

    def set_final_rect_data(self, rect_data_t: dict, vertices: list[tuple[int, int]]):
        """Set the final free-space bounding rectangle data and vertices."""
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        # vertices: [(x, y), ...]
        self._final_rect_data = rect_data_t
        self.final_vertices = vertices
        self.final_rectangle = Rectangle(self.final_rect_data)

    def set_map_orig(self, map_orig: np.ndarray):
        self.map_orig = map_orig

    def set_map_ws(self, map_ws: np.ndarray):
        self.map_ws = map_ws
        self.map_ws_segmented = map_ws.copy()

    def set_map_sf_orig(self, map_sf_orig: np.ndarray):
        self.map_sf_orig = map_sf_orig

    def set_map_cleaning(self, map_cleaning: np.ndarray):
        self.map_cleaning = map_cleaning

    def set_map_viz(self, map_viz: np.ndarray):
        self.map_planning = map_viz
        # # change color of all non-free space to gray for better visualization
        # map_viz_gray = cv.cvtColor(map_viz, cv.COLOR_BGR2GRAY)
        # mask = (map_viz_gray != 255)
        # map_viz[mask] = [200, 200, 200]
        # self.map_planning = map_viz

    def set_hw_bd_px(self, hw_bd: dict):
        """Set pixel bounds and derive local shape and global offset.

        Args:
        - hw_bd: Dict with keys ``h_min``, ``w_min``, ``h_max``, ``w_max``.
        """
        self.hw_bd = hw_bd
        self.hw = (hw_bd["h_max"] - hw_bd["h_min"],
                   hw_bd["w_max"] - hw_bd["w_min"])
        self.global_xy = np.array([hw_bd["w_min"], hw_bd["h_min"]], dtype=np.int32)

    # --------------------------------------------------------------------------
    # Save Data
    # --------------------------------------------------------------------------

    def save_data(self) -> None:
        """Persist maps (PNG) and a JSON metadata file to disk."""
        return  # skip saving for now

        # Save map images with maximum quality (lossless PNG)
        png_params = [cv.IMWRITE_PNG_COMPRESSION, 0]
        cv.imwrite(self.maps_dir + "/" + "local_map_1_orig.png", self.map_orig, png_params)
        cv.imwrite(self.maps_dir + "/" + "local_map_2_sf_orig.png", self.map_sf_orig, png_params)
        cv.imwrite(self.maps_dir + "/" + "local_map_3_cleaning.png", self.map_cleaning, png_params)
        cv.imwrite(self.maps_dir + "/" + "local_map_4_viz.png", self.map_planning, png_params)

        # Save local_map.json
        zone_data_d = dict()
        zone_data_d["origin_m"] = self.orig_map_data.origin_m
        zone_data_d["up_resolution"] = self.orig_map_data.up_resolution
        zone_data_d["hw_bd"] = self.hw_bd
        zone_data_d["rect_data"] = convert_to_native(self.rect_data)
        zone_data_d["final_rect_data"] = convert_to_native(self.final_rect_data)

        with open(self.maps_dir + "/" + "local_map.json", "w") as f:
            json_object = json.dumps(zone_data_d, indent=4)
            f.write(json_object)

    def visualize(self, case: str = "cleaning"):
        if case == "cleaning":
            visualize_map_s(self.map_cleaning, f"zone_{self.ind}_{case}")
        elif case == "viz":
            visualize_map_s(self.map_viz, f"zone_{self.ind}_{case}")
        else:
            visualize_map_s(self.map_orig, f"zone_{self.ind}_{'orig'}")

    def validate_data(self):
        """Placeholder for validating presence/shape of required maps."""
