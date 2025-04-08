""" Events for decomposition """
import math


class Event:
    def __init__(self, obst_ind, ind, v):
        self.ns = "Event"
        self.xy = v
        self.x_max = None
        self.y_max_lim = math.inf
        # self.y_min_lim = -math.inf
        #
        self.etype: str = None      # ceil, floor, updown
        self.etype_2: str = None    # out, in
        self.etype_3: str = None    # ovex, ivex, ocave, icave
        self.c_etype: str = None    # ceil, floor, updown
        self.name = 'event'
        #
        self.obst_ind: int = obst_ind
        self.ind: int = ind         # index in the obstacle (ccw)
        self.s_ind: int = None      # sorted index in the obstacles events
        self.sd_ind: int = None     # sorted index in the decomposition
        self.prev_ind: int = None
        self.next_ind: int = None
        #
        self.hit_up_y: float = None
        self.hit_down_y: float = None
        self.hit_up_event: Event = None
        self.hit_down_event: Event = None
        #
        self.end_up_ed_xy: tuple[float, float] = None
        self.end_down_ed_xy: tuple[float, float] = None
        #
        self.endpoint_cil = []
        self.has_adj = True
        #
        self.fake: bool = False
        self.vertical: bool = False
        #
        #
        self.skip_sweep = False
        self.skip_hit = False
        self.direct_trap = False
        #
        self.done = False
        self.can_up = True
        self.can_down = True
        #
        self.concave = False
        self.convex = True

    def set_s_ind(self, ind: int):
        self.s_ind = ind

    def set_sd_ind(self, ind: int):
        self.sd_ind = ind

    def set_prev_ind(self, ind: int):
        self.prev_ind = ind

    def set_next_ind(self, ind: int):
        self.next_ind = ind

    def set_endpoint_up_edge(self, xy: tuple[float, float]):
        self.end_up_ed_xy = xy

    def set_endpoint_down_edge(self, xy: tuple[float, float]):
        self.end_down_ed_xy = xy

    def add_endpoint_ci(self, ci: str):
        self.endpoint_cil.append(ci)

    def update_name(self):
        name = f"{self.sd_ind}"
        if self.direct_trap:
            name += "-direct"
        elif self.etype_3:
            name += f"-{self.etype_3}"
        elif self.etype_2:
            name += f"-{self.etype_2}"
        elif self.etype:
            name += f"-{self.etype}"
        if self.fake:
            name += "-fake"
        if self.skip_sweep:
            name += "-sk"
        if self.skip_hit:
            name += "-h"
        self.name = name
