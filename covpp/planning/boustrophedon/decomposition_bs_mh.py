"""New Trapezoidal decomposition module for boustrophedon cell generation."""
import math
from shapely import Polygon
from covpp.base.logger import MyLogger
from covpp.base.configs import PlanningConfig
from covpp.planning.boustrophedon.events import Event
from covpp.planning.boustrophedon.obstacles import Obstacle
from covpp.planning.boustrophedon.cell import Cell, CellGroup
from covpp.planning.boustrophedon.plotter import BoustPlotter
from covpp.planning.boustrophedon.post_process_mcells import eval_cells_path_shift


class CellDecomposition:
    """Perform new trapezoidal decomposition for boustrophedon path planning.

    This class decomposes a workspace into trapezoidal cells suitable for
    parallel sweeping coverage patterns, handles cell adjacency, and merges
    compatible cells for efficient coverage.

    Decomposition Pipeline:
        1. Create obstacles and events
        2. Decompose into trapezoidal cells
        3. Find adjacent cells
        4. Complete adjacency relationships
        5. Create cell groups and merge compatible cells
        6. Set merged cell adjacency
        7. Evaluate cells for planning requirements
        8. Visualize decomposition results
    """

    def __init__(self, resolution: float = 0.05, result_dir: str = None):
        self.lg = MyLogger("CellDecompositionMH")
        #
        self.result_dir: str = result_dir
        self.resolution: float = resolution
        self.bcfgs = PlanningConfig()
        #
        self.b_poly: Polygon = None
        self.b_obstacle: Obstacle = None
        self.polygons: list[Polygon] = []
        #
        self.obstacles: list[Obstacle] = []
        self.events: list[Event] = []
        self.cells: list[Cell] = []
        self.trapezoids: list[Polygon] = []
        self.n_events: int = 0
        self.n_cells: int = 0
        #
        self.n_mcells: int = 0
        self.cells_groups: list[CellGroup] = []
        self.mcells: list[Cell] = []
        #
        self.plotter: BoustPlotter = None

    def initiate(self):
        """Execute complete decomposition pipeline.

        Runs all steps from obstacle creation through visualization of the
        final decomposed cells.
        """
        self.create_obstacles()
        self.create_events()

        #
        self.decompose_all()
        self.find_adjacents_cells()
        self.complete_adjacency()

        # create merged cells & set_merged_cells_adjacents
        self.create_cell_groups()
        self.set_merged_cells_adjacents()
        self.n_mcells = len(self.mcells)

        # find if coverage pp in cell should be shifted based on boarder with adjacents
        eval_cells_path_shift(self.mcells)

        #
        self.eval_mcells()
        self.visualize()

    def set_bounding_polygon(self, poly: Polygon):
        """Set the workspace boundary polygon."""
        self.b_poly = poly
        self.b_obstacle = Obstacle(0, self.b_poly)

    def set_obstacle_polygons(self, polygons: list[Polygon]):
        """Set the list of obstacle polygons within the workspace."""
        self.polygons = polygons

    def create_obstacles(self):
        self.obstacles.append(self.b_obstacle)
        for i, p in enumerate(self.polygons):
            self.obstacles.append(Obstacle(i+1, p))

    def create_events(self):
        all_events = [event for obstacle in self.obstacles for event in obstacle.events]
        self.n_events = len(all_events)
        self.events = sorted(all_events, key=lambda e: (e.xy))
        for i, e in enumerate(self.events):
            e.set_decomp_sorted_ind(i)
            e.update_name()

    def add_cell(self, poly: Polygon, event: Event):
        self.trapezoids.append(poly)
        cell = Cell(self.n_cells, poly, event, self.resolution)
        self.cells.append(cell)
        self.n_cells += 1

    # decompose_all ====================================================================================================

    def decompose_all(self):
        """
        Decompose all the events to create trapezoidal cells.

        Steps:
            1- Iterate over all events (skip skip_sweep and done events).
            2- For each event, find hits by sweeping line (up and down) (sweep_up_down).
            3- Evaluate done status of hit events (eval_done_status).
            4- Create trapezoids based on event type and hits (create_trapezoid).
        """
        for i, base_e in enumerate(self.events[1:]):
            ind = i + 1

            # Skip border and events
            if base_e.skip_sweep or base_e.done:
                continue

            # find hits of the sweeping line at the base event
            hit_up_y, hit_down_y, hit_up_event, hit_down_event = self.sweep_up_down(base_e, ind)

            # Update base event with hit results
            base_e.hit_up_y, base_e.hit_up_event = hit_up_y, hit_up_event
            base_e.hit_down_y, base_e.hit_down_event = hit_down_y, hit_down_event

            # Update done status of the hit events
            self.eval_done_status(base_e, hit_up_event, hit_down_event)

            # Create trapezoids based on event type
            self.create_trapezoid(base_e, hit_up_event, hit_down_event)

    def sweep_up_down(self, base_e: Event, base_i: int):
        """
        Sweep line from the base event to find hit events.

        Method:
         1- Iterate over events after the base event.
         2- For each event, evaluate if it can be hit (eval_event_up and eval_event_down).
         3- Update hit variables if successful.
         4- Return hit results.
        """

        hit_up_y, hit_down_y = math.inf, -math.inf
        hit_up_event, hit_down_event = None, None

        # check if the event needs direct trapezoid
        if base_e.direct_trap:
            return hit_up_y, hit_down_y, hit_up_event, hit_down_event

        # Iterate over events
        for e in self.events[base_i+1:]:
            if e.skip_hit:
                continue

            # Evaluate event based on base_e type
            success_up, y_up = False, None
            success_down, y_down = False, None

            if base_e.etype1_updown in ('up', 'updown'):
                success_up, y_up = self.eval_event_up(base_e, e, hit_up_y)
            if base_e.can_down and base_e.etype1_updown in ('down', 'updown'):
                success_down, y_down = self.eval_event_down(base_e, e, hit_down_y)

            # Update hit variables if successful
            if success_up:
                hit_up_y, hit_up_event = y_up, e
            if success_down:
                hit_down_y, hit_down_event = y_down, e
        #
        return hit_up_y, hit_down_y, hit_up_event, hit_down_event

    def eval_event_up(self, base_e: Event, e: Event, y_hit: float) -> bool:
        """
        Evaluate if event e can be hit when sweeping up from base_e.

        Method:
        1- Check if e is of type 'up'
        2- Check if e is not hit from outside of polygon
        3- Check for higher 'updown' events with same x coordinate
        4- Evaluate if sweep line hits the edge formed by e and its previous event based on coordinates
        5- Return success status and new hit y-coordinate if successful

        """
        if e.etype1_updown == 'up':
            return False, None

        # check not to hit from outside of polygon
        if e.direct_trap and base_e.xy[0] > e.x_max:
            return False, None

        # reach to higher updown event
        new_y_hit = e.xy[1]
        if e.xy[0] == base_e.xy[0] and new_y_hit < y_hit:
            if e.etype1_updown == 'updown':
                if e.etype2_inout == 'out':
                    return True, new_y_hit
                # in
                return False, None

        # type is 'down'
        # edge = e1-e
        e1 = e.prev_e

        # check coordinates
        # if e1 is the base event itself
        if e1.dec_s_ind == base_e.dec_s_ind:
            return False, None
        # if x base_e is left to e1
        if e1.xy[0] >= base_e.xy[0]:
            return False, None
        # check y coordinates
        if e1.xy[1] < base_e.xy[1] and e.xy[1] < base_e.xy[1]:
            return False, None

        # incline
        theta = math.atan2(e.xy[1] - e1.xy[1], e.xy[0] - e1.xy[0])
        new_y_hit = math.tan(theta) * (base_e.xy[0] - e1.xy[0]) + e1.xy[1]
        if new_y_hit < y_hit:
            return True, new_y_hit
        #
        return False, None

    def eval_event_down(self, base_e: Event, e: Event, y_hit: float) -> bool:
        """
        Evaluate if event e can be hit when sweeping down from base_e.

        Method:
        1- Check if e is of type 'down'
        2- Evaluate if sweep line hits the edge formed by e and its next event based on coordinates
        3- Return success status and new hit y-coordinate if successful
        """

        if e.etype1_updown == 'down':
            return False, None

        # type is 'up'
        # edge = e1-e
        e2 = e.next_e

        # check coordinates
        # if e2 is the base event itself
        if e2.dec_s_ind == base_e.dec_s_ind:
            return False, None
        # if x base_e is left to e2
        if e2.xy[0] >= base_e.xy[0]:
            return False, None
        # check y coordinates
        if e2.xy[1] > base_e.xy[1] and e.xy[1] > base_e.xy[1]:
            return False, None

        # incline
        theta = math.atan2(e.xy[1] - e2.xy[1], e.xy[0] - e2.xy[0])
        new_y_hit = math.tan(theta) * (base_e.xy[0] - e2.xy[0]) + e2.xy[1]
        if new_y_hit > y_hit:
            return True, new_y_hit
        #
        return False, None

    def eval_done_status(self, base_e: Event, hit_up_event: Event, hit_down_event: Event):
        """
        Evaluate done status of hit events.

        Method:
        1- Evaluate events on the same x coordinate between base_e and hit_up_event
        2- Update done status of the hit events
        """
        # eval events on the same x coordinate between base_e and hit_up_event
        if hit_up_event:
            up_ind = base_e.dec_s_ind+1
            up_e = self.events[up_ind]
            while up_e.xy[0] == base_e.xy[0] and up_e.xy[1] < base_e.hit_up_y:
                up_e.done = True
                up_ind = up_ind + 1
                up_e = self.events[up_ind]

        # update done status of the hit events
        if hit_up_event:
            if hit_up_event.xy[0] == base_e.xy[0]:
                if hit_up_event.etype1_updown == 'down':
                    hit_up_event.done = True
                elif hit_up_event.etype1_updown == 'updown':
                    hit_up_event.can_down = False
        if hit_down_event:
            if hit_down_event.xy[0] == base_e.xy[0]:
                if hit_down_event.etype1_updown == 'up':
                    hit_down_event.done = True

    def create_trapezoid(self, base_e: Event, hit_up_event: Event, hit_down_event: Event):
        """
        Create trapezoids based on event type and hits.

        Method:
        1- Handle direct trap case.
        2- Handle not hit events case.
        3- Create trapezoids based on event type (updown, up, down).
        4- Add cell number and update endpoints of hit events.
        5- Create cell(s) and add to the list.

        Trapezoid vertex are: base_e, base_e_hit (up or down), hit_event endpoint, base_e endpoint
        """
        trap_up, trap_down = None, None

        # direct trap
        if base_e.direct_trap:
            trap1 = [base_e.xy, base_e.end_down_ed_xy, base_e.end_up_ed_xy]
            base_e.set_base_ci_up(self.n_cells)
            self.add_cell(Polygon(trap1), base_e)
            return

        # not hit events
        if not (hit_up_event or hit_down_event):
            return

        hit_up_y = base_e.hit_up_y
        hit_down_y = base_e.hit_down_y

        # updown
        if base_e.etype1_updown == 'updown':
            if base_e.etype2_inout == 'out':
                if hit_up_event:
                    trap_up = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy, base_e.end_up_ed_xy]
                if hit_down_event:
                    trap_down = [base_e.xy, (base_e.xy[0], hit_down_y), hit_down_event.end_up_ed_xy, base_e.end_down_ed_xy]
            else:
                if hit_down_event and hit_up_event:
                    trap_down = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy,
                                 hit_down_event.end_up_ed_xy, (base_e.xy[0], hit_down_y)]
                else:
                    self.lg.warn("[create_trapezoid]: updown: hit_up_event and hit_down_event are both None [updown-out]")
                    base_e.hit_down_event = None
                    base_e.hit_up_event = None
                    return
            if not (hit_up_event or hit_down_event):
                self.lg.warn("[create_trapezoid]: updown: hit_up_event and hit_down_event are both None")

        # up
        elif base_e.etype1_updown == 'up' and hit_up_event:
            trap_up = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy, base_e.end_up_ed_xy]
        # down
        elif base_e.etype1_updown == 'down' and hit_down_event:
            trap_down = [base_e.xy, base_e.end_down_ed_xy, hit_down_event.end_up_ed_xy, (base_e.xy[0], hit_down_y)]
        else:
            self.lg.warn("[create_trapezoid]: Invalid trapezoid creation conditions.")
            raise ValueError

        # add cell number - update endpoints
        dn = 0
        if hit_up_event:
            dn = 1
            if base_e.etype2_inout == 'in':
                dn = 0
            base_e.set_base_ci_up(self.n_cells)
            hit_up_event.add_down_ci_d(base_e.dec_s_ind, self.n_cells)
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
        if hit_down_event:
            base_e.set_base_ci_down(self.n_cells+dn)
            hit_down_event.add_up_ci_d(base_e.dec_s_ind, self.n_cells+dn)
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))

        # create cell
        if trap_up:
            self.add_cell(Polygon(trap_up), base_e)
        if trap_down:
            self.add_cell(Polygon(trap_down), base_e)
        if trap_up is None and trap_down is None:
            self.lg.error("[create_trapezoid]: Invalid trapezoid creation")

    # find_adjacents_cells =============================================================================================

    def find_adjacents_cells(self):
        """
        Find adjacent cells based on hit events.

        Method:
        1- Iterate over all events.
        2- Find adjacents based on hit_up_event and hit_down_event.
        3- Update adjacent and parent cell indices for the involved cells.
        """

        for e in self.events:
            up_adj_ci, down_adj_ci = None, None
            current_cell: Cell = None

            # hit_up_event
            if e.hit_up_event:
                hit_up_event = e.hit_up_event
                ci_curr = e.base_ci_up
                current_cell = self.cells[ci_curr]
                ind = hit_up_event.down_cil_k.index(e.dec_s_ind)
                if len(hit_up_event.down_cil_k) > ind+1:
                    # next cell created on hit_up_event
                    val = e.hit_up_event.down_cil_k[ind+1]
                    up_adj_ci = e.hit_up_event.down_cil_d[val]
                elif hit_up_event.direct_trap:
                    up_adj_ci = hit_up_event.base_ci_up
                else:
                    # e was the last hitting event
                    up_adj_ci = hit_up_event.base_ci_down
                if up_adj_ci:
                    current_cell.set_up_adj_ci(up_adj_ci)
                    self.cells[up_adj_ci].set_parent_up_ci(ci_curr)

            # hit_down_event
            if e.hit_down_event:
                ci_curr = e.base_ci_down
                current_cell = self.cells[ci_curr]
                ind = e.hit_down_event.up_cil_k.index(e.dec_s_ind)
                if len(e.hit_down_event.up_cil_k) > ind+1:
                    val = e.hit_down_event.up_cil_k[ind+1]
                    down_adj_ci = e.hit_down_event.up_cil_d[val]
                else:
                    down_adj_ci = e.hit_down_event.base_ci_up
                if down_adj_ci:
                    current_cell.set_down_adj_ci(down_adj_ci)
                    self.cells[down_adj_ci].set_parent_down_ci(ci_curr)

    def complete_adjacency(self):
        """
        Complete adjacency by checking geometric conditions.

        Method:
        1- Iterate over all cells.
        2- For each cell, check for potential adjacent cells on the right side.
            - Compare x_max of the current cell with x_min of the next cell.
            - Check y-overlap conditions.
        3- If conditions are met, add adjacent and parent relationships.
        """

        cell: Cell = None
        for ic, cell in enumerate(self.cells):
            e = cell.event
            # adjacent cells
            if e.has_adj:
                for next_c in self.cells[ic+1:]:
                    #
                    if next_c.id in cell.mid_adjlist:
                        continue
                    #
                    if cell.x_max_p == next_c.x_min_p:
                        # check y condition
                        if next_c.y_min_p < cell.y_mid_p < next_c.y_max_p or \
                                cell.y_min_p < next_c.y_mid_p < cell.y_max_p:
                            # add adjacent cells
                            cell.add_mid_adjlist(next_c.id)
                            next_c.add_mid_parentlist(cell.id)
                #
                if cell.n_adj == 0:
                    cell.found_connection = False
                    self.lg.error("[find_adjacents_cells]: No adjacents found:")
                    # self.lg.error(f"e: {cell.event.dec_s_ind}, -- cell: {cell.id}")

            if e.has_parent:
                if cell.n_parent == 0:
                    cell.found_connection = False
                    self.lg.error("[find_adjacents_cells]: No parents found:")
                    # self.lg.error(f"e: {cell.event.dec_s_ind}, -- cell: {cell.id}")

# cell groups and merged cells =========================================================================================

    def create_cell_groups(self):
        """ Create cell groups and merged cells.
        Procedure:
            1- Iterate over all cells.
            2- For each ungrouped cell, create a new group.
            3- If the base cell has multiple adjacents, only add it to the group (no adjacents added).
            4- If the base cell has a single adjacent, traverse through adjacents
               and add them to the group until a cell with multiple adjacents or a grouped cell is reached.
            5- After forming the group, update group properties (first and last cell) and store the merged cell.
        """
        ncg = 0
        cells_groups: list[CellGroup] = []
        mcells: list[Cell] = []
        #
        for cell in self.cells[0:]:
            if cell.grouped:
                continue
            base_cell = cell
            base_cell.grouped = True
            if base_cell.n_adj == 0:
                new_group = CellGroup(base_cell, ncg)
            elif base_cell.n_adj > 1:
                new_group = CellGroup(base_cell, ncg)
            else:
                adj_cell: Cell = None
                if base_cell.up_adj_ci:
                    adj_cell = self.cells[base_cell.up_adj_ci]
                elif base_cell.down_adj_ci:
                    adj_cell = self.cells[base_cell.down_adj_ci]
                else:
                    adj_cell = self.cells[base_cell.mid_adjlist[0]]

                #
                new_group = CellGroup(base_cell, ncg)
                while adj_cell.n_parent == 1 and not adj_cell.grouped:
                    adj_cell.grouped = True
                    new_group.add_cell(adj_cell)
                    #
                    if not adj_cell.n_adj == 1:
                        break
                    #
                    if adj_cell.up_adj_ci:
                        adj_cell = self.cells[adj_cell.up_adj_ci]
                    elif adj_cell.down_adj_ci:
                        adj_cell = self.cells[adj_cell.down_adj_ci]
                    else:
                        adj_cell = self.cells[adj_cell.mid_adjlist[0]]
            # merge_cells
            if new_group:
                new_group.update_group()
                cells_groups.append(new_group)
                mcells.append(new_group.merged_cell)
                ncg += 1
        #
        self.cells_groups = cells_groups
        self.mcells = mcells

    def set_merged_cells_adjacents(self):
        """ Set merged cells adjacents based on cell groups (first and last cell). """
        for cellg in self.cells_groups:
            parents_i = cellg.first_cell.mid_parentlist
            successors_i = cellg.last_cell.mid_adjlist
            cellg.merged_cell.mid_parentlist = [self.cells[i].group_id for i in parents_i]
            cellg.merged_cell.mid_adjlist = [self.cells[i].group_id for i in successors_i]
            adjs = cellg.merged_cell.mid_parentlist.copy()
            adjs.extend(cellg.merged_cell.mid_adjlist)
            cellg.merged_cell.adjacent_ids = adjs

    # ==================================================================================================================

    def eval_mcells(self):
        """check if area of the cell is enough for planning. """
        for cell in self.mcells:
            cell.eval_planning_area(self.resolution, self.bcfgs.cleaning_r)

    def visualize(self):
        self.plotter = BoustPlotter(self)
        self.plotter.visualize()
