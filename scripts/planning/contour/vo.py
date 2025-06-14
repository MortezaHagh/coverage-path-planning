from scipy.spatial import Voronoi, voronoi_plot_2d
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.map_polygons import MapPolygons


class MyVoronoi:
    def __init__(self, zone_map_data: ZoneMapData, map_polygons: MapPolygons):
        self.ns = "Voronoi"
        print(f"[{self.ns}]: Initializing Voronoi diagram...")

        self.zone_map_data = zone_map_data
        self.map_polygons = map_polygons
        self.voronoi = None

    def get_voronoi(self):
        points = self.zone_map_data.get_points()
        self.voronoi = Voronoi(points)
        if self.voronoi is None:
            raise ValueError("GEtting Voronoi failed!")

    def plot_voronoi(self):
        if self.voronoi is None:
            raise ValueError(f"[{self.ns}]: Voronoi diagram has not been computed yet. Call get_voronoi() first.")

        voronoi_plot_2d(self.voronoi, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6)
        self.map_polygons.plot_map()
        self.zone_map_data.plot_points()
