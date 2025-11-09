""" Event class for decomposition """


class Event:
    """Event class for decomposition."""

    def __init__(self, obst_ind: int, ind: int, v: tuple[float, float]):
        self.xy: tuple[float, float] = v
        self.x_max: float = None
        # self.y_max_lim: float = math.inf
        # self.y_min_lim: float = -math.inf
        #
        self.name: str = 'event'            # event name ( will be updated later based on type and index)
        self.etype1_updown: str = None      # up, down, updown
        self.etype2_inout: str = None       # out, in
        self.etype3_io_conc: str = None     # ovex, ivex, ocave, icave
        #
        self.obst_ind: int = obst_ind       # index in the obstacle to which this event belongs
        self.ind: int = ind                 # index in the obstacle (ccw)
        self.obs_s_ind: int = None          # sorted index in the obstacle events
        self.dec_s_ind: int = None          # sorted index in the decomposition (all) events
        self.next_e: 'Event' = None         # next event in the obstacle
        self.prev_e: 'Event' = None         # previous event in the obstacle
        #
        self.hit_up_y: float = None         # y of the event which this event hit when sweeping up
        self.hit_down_y: float = None       # y of the event which this event hit when sweeping down
        self.hit_up_event: 'Event' = None   # event which this event hit when sweeping up
        self.hit_down_event: 'Event' = None  # event which this event hit when sweeping down

        # end edge xy (up and down)
        self.end_up_ed_xy: tuple[float, float] = None       # endpoint of the up edge
        self.end_down_ed_xy: tuple[float, float] = None     # endpoint of the down edge

        # ci: cell index, cil: cell index list
        self.base_ci_up: int = None         # cell index which created from this event by hit up
        self.base_ci_down: int = None       # cell index which created from this event by hit down
        self.up_cil_d = dict()              # cell index list dictionary - created by other hit down event which hitted to this event
        self.up_cil_k = []                  # keys list of up_cil_d
        self.down_cil_d = dict()            # cell index list dictionary - created by other hit up event which hitted to this event
        self.down_cil_k = []                # keys list of down_cil_d
        #
        self.has_adj: bool = True           # if this event should has adjacent cells
        self.has_parent: bool = True        # if this event should has parent cells
        #
        self.vertical: bool = False         # is in vertical edge (set in the obstacle processing)
        self.fake: bool = False             # is a fake event  (added between two vertical vertices) (set in the obstacle processing)
        #
        #
        self.skip_sweep: bool = False       # skip sweeping from this event (set in the obstacle processing)
        self.skip_hit: bool = False
        self.direct_trap: bool = False
        #
        self.done: bool = False         # no need to do sweep and hit for this event (set during decomposition)
        self.can_down: bool = True      # do sweep down from this event (set during decomposition)
        # self.can_up: bool = True
        #
        self.concave: bool = False
        self.convex: bool = True

    def set_sorted_ind(self, ind: int):
        self.obs_s_ind = ind

    def set_decomp_sorted_ind(self, ind: int):
        self.dec_s_ind = ind

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
        name = f"{self.dec_s_ind}"
        if self.direct_trap:
            name += "-direct"
        elif self.etype3_io_conc:
            name += f"-{self.etype3_io_conc}"
        elif self.etype2_inout:
            name += f"-{self.etype2_inout}"
        elif self.etype1_updown:
            name += f"-{self.etype1_updown}"
        if self.fake:
            name += "-fake"
        if self.skip_sweep:
            name += "-sk"
        if self.skip_hit:
            name += "-h"
        self.name = name
