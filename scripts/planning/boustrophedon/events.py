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
        self.name = 'event'
        self.etype: str = None      # up, down, updown
        self.etype_2: str = None    # out, in
        self.etype_3: str = None    # ovex, ivex, ocave, icave
        #
        self.obst_ind: int = obst_ind   # index in the obstacle to which this event belongs
        self.ind: int = ind             # index in the obstacle (ccw)
        self.s_ind: int = None          # sorted index in the obstacle events
        self.sd_ind: int = None         # sorted index in the decomposition (all) events
        self.next_e: 'Event' = None     # next event in the obstacle
        self.prev_e: 'Event' = None     # previous event in the obstacle
        #
        self.hit_up_y: float = None
        self.hit_down_y: float = None
        self.hit_up_event: 'Event' = None
        self.hit_down_event: 'Event' = None

        # end edge xy (up and down)
        self.end_up_ed_xy: tuple[float, float] = None       # updated by hit down event
        self.end_down_ed_xy: tuple[float, float] = None     # updated by hit up event

        # ci: cell index, cil: cell index list
        self.base_ci_up = None        # cell index which created from this event by hit up
        self.base_ci_down = None      # cell index which created from this event by hit down
        self.up_cil_d = dict()        # cell index list dictionary - created by other hit down event which hitted to this event
        self.up_cil_k = []            # keys list of up_cil_d
        self.down_cil_d = dict()      # cell index list dictionary - created by other hit up event which hitted to this event
        self.down_cil_k = []          # keys list of down_cil_d
        #
        self.has_adj = True         # if this event should has adjacent cells
        self.has_parent = True      # if this event should has parent cells
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
        self.can_down = True
        # self.can_up = True
        #
        self.concave = False
        self.convex = True

    def set_s_ind(self, ind: int):
        self.s_ind = ind

    def set_sd_ind(self, ind: int):
        self.sd_ind = ind

    def set_prev_e(self, e: 'Event'):
        self.prev_e = e

    def set_next_e(self, e: 'Event'):
        self.next_e = e

    def set_endpoint_up_edge(self, xy: tuple[float, float]):
        self.end_up_ed_xy = xy

    def set_endpoint_down_edge(self, xy: tuple[float, float]):
        self.end_down_ed_xy = xy

    # ==========================================================================

    def set_base_ci_up(self, ci):
        self.base_ci_up = ci

    def set_base_ci_down(self, ci):
        self.base_ci_down = ci

    def add_up_ci_d(self, base_ind: int, ci):
        self.up_cil_d[base_ind] = ci
        self.up_cil_k.append(base_ind)

    def add_down_ci_d(self, base_ind: int, ci):
        self.down_cil_d[base_ind] = ci
        self.down_cil_k.append(base_ind)

    # ==========================================================================

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
