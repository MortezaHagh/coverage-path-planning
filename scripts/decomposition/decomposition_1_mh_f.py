""" Trapezoidal decomposition """
import math
from shapely import Polygon
from scripts.decomposition.events import Event
from scripts.decomposition.obstacles import Obstacle
from scripts.decomposition.cell import Cell, CellGroup
from scripts.decomposition.plotter import BoustPlotter
from scripts.decomposition.optimize_mcells_seq import OptimCellsSeq
from scripts.base.configs import DistrictConfgs


class MHDecomposition:
    def __init__(self):
        self.ns = "MHDecomposition"
        self.dcfgs = DistrictConfgs()
        self.b_poly: Polygon = None
        self.b_obstacle: Obstacle = None
        self.polygons: list[Polygon] = []
        self.obstacles: list[Obstacle] = []
        self.events: list[Event] = []
        self.cells: list[Cell] = []
        self.trapezoids: list[Polygon] = []
        self.n_events: int = 0
        self.n_cells: int = 0
        #
        self.cells_groups: list[CellGroup] = []
        self.mcells: list[Cell] = []
        #
        self.plotter: BoustPlotter = None

    def initiate(self):
        self.create_obstacles()
        self.create_events()
        #
        self.decompose_all()
        self.find_adjacents_cells()
        self.compleate_adjacency()
        self.create_cell_groups()
        self.set_merged_cells_adjacents()
        self.create_sequence()
        self.visualize()
        #

    def set_bounding_polygon(self, poly: Polygon):
        self.b_poly = poly
        self.b_obstacle = Obstacle(0, self.b_poly)

    def set_obstacle_polygons(self, polygons: list[Polygon]):
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
            e.set_sd_ind(i)
            e.update_name()

    def add_cell(self, poly: Polygon, event: Event):
        self.trapezoids.append(poly)
        cell = Cell(self.n_cells, poly, event)
        self.cells.append(cell)
        self.n_cells += 1

    # decompose_all ====================================================================================================

    def decompose_all(self):
        for i, base_e in enumerate(self.events[1:]):
            ind = i + 1

            # Skip border and events
            if base_e.skip_sweep or base_e.done:
                continue

            # find hits of the sweeping line at the base event
            hit_up_y, hit_down_y, hit_up_event, hit_down_event = self.expand_hit(base_e, ind)

            # Update base event with hit results
            base_e.hit_up_y, base_e.hit_up_event = hit_up_y, hit_up_event
            base_e.hit_down_y, base_e.hit_down_event = hit_down_y, hit_down_event

            # Update done status of the hit events
            self.eval_done_status(base_e, hit_up_event, hit_down_event)

            # Create trapezoids based on event type
            self.create_trapezoid(base_e, hit_up_event, hit_down_event)

    def expand_hit(self, base_e: Event, base_i: int):
        # Initialize hit variables
        hit_up_y, hit_down_y = math.inf, -math.inf
        hit_up_event, hit_down_event = None, None

        # check if the event needs direct trapezoid
        if base_e.direct_trap:
            return hit_up_y, hit_down_y, hit_up_event, hit_down_event

        # Iterate over events
        for e in self.events[base_i+1:]:
            #
            if e.skip_hit:
                continue

            # Evaluate event based on base_e type
            success_up, y_up = False, None
            success_down, y_down = False, None

            if base_e.etype in ('up', 'updown'):
                success_up, y_up = self.eval_event_up(base_e, e, hit_up_y)
            if base_e.can_down and base_e.etype in ('down', 'updown'):
                success_down, y_down = self.eval_event_down(base_e, e, hit_down_y)

            # Update hit variables if successful
            if success_up:
                hit_up_y, hit_up_event = y_up, e
            if success_down:
                hit_down_y, hit_down_event = y_down, e
        #
        return hit_up_y, hit_down_y, hit_up_event, hit_down_event

    def eval_event_up(self, base_e: Event, e: Event, y_hit: float) -> bool:
        # based even type is 'up'
        if e.etype == 'up':
            return False, None

        # check not to hit from outside of polygon
        if e.direct_trap and base_e.xy[0] > e.x_max:
            return False, None

        # reach to higher updown event
        new_y_hit = e.xy[1]
        if e.xy[0] == base_e.xy[0] and new_y_hit < y_hit:
            if e.etype == 'updown':
                if e.etype_2 == 'out':
                    return True, new_y_hit
                # in
                return False, None

        # type is 'down'
        # edge = e1-e
        e1 = e.prev_e

        # check coordinates
        # if e1 is the base event itself
        if e1.sd_ind == base_e.sd_ind:
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
        # based even type is 'down'
        if e.etype == 'down':
            return False, None

        # type is 'up'
        # edge = e1-e
        e2 = e.next_e

        # check coordinates
        # if e2 is the base event itself
        if e2.sd_ind == base_e.sd_ind:
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
        # eval events on the same x coordinate between base_e and hit_up_event
        if hit_up_event:
            up_ind = base_e.sd_ind+1
            up_e = self.events[up_ind]
            while up_e.xy[0] == base_e.xy[0] and up_e.xy[1] < base_e.hit_up_y:
                up_e.done = True
                up_ind = up_ind + 1
                up_e = self.events[up_ind]

        # update done status of the hit events
        if hit_up_event:
            if hit_up_event.xy[0] == base_e.xy[0]:
                if hit_up_event.etype == 'down':
                    hit_up_event.done = True
                elif hit_up_event.etype == 'updown':
                    hit_up_event.can_down = False
        if hit_down_event:
            if hit_down_event.xy[0] == base_e.xy[0]:
                if hit_down_event.etype == 'up':
                    hit_down_event.done = True

    def create_trapezoid(self, base_e: Event, hit_up_event: Event, hit_down_event: Event):
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
        if base_e.etype == 'updown':
            if base_e.etype_2 == 'out':
                if hit_up_event:
                    trap_up = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy, base_e.end_up_ed_xy]
                if hit_down_event:
                    trap_down = [base_e.xy, (base_e.xy[0], hit_down_y), hit_down_event.end_up_ed_xy, base_e.end_down_ed_xy]
            else:
                if hit_down_event and hit_up_event:
                    trap_down = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy,
                                 hit_down_event.end_up_ed_xy, (base_e.xy[0], hit_down_y)]
                else:
                    print((f"[{self.ns}][create_trapezoid][WARN]: hit_up_event and hit_down_event are both None [updown-in]"))
                    return
            if not (hit_up_event or hit_down_event):
                print((f"[{self.ns}][create_trapezoid][WARN]: updown: hit_up_event and hit_down_event are None"))

        # up
        elif base_e.etype == 'up' and hit_up_event:
            trap_up = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy, base_e.end_up_ed_xy]
        # down
        elif base_e.etype == 'down' and hit_down_event:
            trap_down = [base_e.xy, base_e.end_down_ed_xy, hit_down_event.end_up_ed_xy, (base_e.xy[0], hit_down_y)]

        # add cell number - update endpoints
        dn = 0
        if hit_up_event:
            dn = 1
            if base_e.etype_2 == 'in':
                dn = 0
            base_e.set_base_ci_up(self.n_cells)
            hit_up_event.add_down_ci_d(base_e.sd_ind, self.n_cells)
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
        if hit_down_event:
            base_e.set_base_ci_down(self.n_cells+dn)
            hit_down_event.add_up_ci_d(base_e.sd_ind, self.n_cells+dn)
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))

        # create cell
        if trap_up:
            self.add_cell(Polygon(trap_up), base_e)
        if trap_down:
            self.add_cell(Polygon(trap_down), base_e)
        if trap_up is None and trap_down is None:
            print((f"[{self.ns}][create_trapezoid][ERROR]: Invalid trapezoid creation"))

    # find_adjacents_cells =============================================================================================

    def find_adjacents_cells(self):
        for e in self.events:
            up_adj_ci, down_adj_ci = None, None
            current_cell: Cell = None

            # hit_up_event
            if e.hit_up_event:
                hit_up_event = e.hit_up_event
                ci_curr = e.base_ci_up
                current_cell = self.cells[ci_curr]
                ind = hit_up_event.down_cil_k.index(e.sd_ind)
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
                ind = e.hit_down_event.up_cil_k.index(e.sd_ind)
                if len(e.hit_down_event.up_cil_k) > ind+1:
                    val = e.hit_down_event.up_cil_k[ind+1]
                    down_adj_ci = e.hit_down_event.up_cil_d[val]
                else:
                    down_adj_ci = e.hit_down_event.base_ci_up
                if down_adj_ci:
                    current_cell.set_down_adj_ci(down_adj_ci)
                    self.cells[down_adj_ci].set_parent_down_ci(ci_curr)

    def compleate_adjacency(self):
        for ic, cell in enumerate(self.cells):
            e = cell.event
            # adjacent cells
            if e.has_adj:
                for next_c in self.cells[ic+1:]:
                    #
                    if next_c.ind in cell.mid_adjlist:
                        continue
                    #
                    if cell.x_max == next_c.x_min:
                        # check y condition
                        if next_c.y_min < cell.y_mid < next_c.y_max or \
                                cell.y_min < next_c.y_mid < cell.y_max:
                            # add adjacent cells
                            cell.add_mid_adjlist(next_c.ind)
                            next_c.add_mid_parentlist(cell.ind)
                #
                if cell.n_adj == 0:
                    cell.status = False
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: No adjacents found:")
                    print(f"e: {cell.event.sd_ind}, -- cell: {cell.ind}")

            if e.has_parent:
                if cell.n_parent == 0:
                    cell.status = False
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: No parents found:")
                    print(f"e: {cell.event.sd_ind}, -- cell: {cell.ind}")

    def create_cell_groups(self):
        ncg = 0
        for cell in self.cells[0:]:
            if cell.done:
                continue
            base_cell = cell
            base_cell.done = True
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
                while adj_cell.n_parent == 1 and not adj_cell.done:
                    adj_cell.done = True
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
                self.cells_groups.append(new_group)
                self.mcells.append(new_group.merged_cell)
                ncg += 1

    def set_merged_cells_adjacents(self):
        for cellg in self.cells_groups:
            parents_i = cellg.first_cell.mid_parentlist
            successors_i = cellg.last_cell.mid_adjlist
            cellg.merged_cell.mid_parentlist = [self.cells[i].group_ind for i in parents_i]
            cellg.merged_cell.mid_adjlist = [self.cells[i].group_ind for i in successors_i]
            adj = cellg.merged_cell.mid_parentlist.copy()
            adj.extend(cellg.merged_cell.mid_adjlist)
            cellg.merged_cell.adjacent_inds = adj

    # =================================================================================================================

    def create_sequence(self):
        if not self.dcfgs.optimize_cell_path:
            for i, cell in enumerate(self.mcells):
                next_cell_ind = None
                if i < len(self.mcells) - 1:
                    next_cell_ind = self.mcells[i + 1].ind
                cell.set_optim_ind(i, next_cell_ind)
            return
        OptimCellsSeq(self.mcells)

    # =================================================================================================================

    def get_event(self, ind: int) -> Event:
        if ind < 0 or ind >= len(self.events):
            return None
        return self.events[ind]

    def visualize(self):
        self.plotter = BoustPlotter(self)
        self.plotter.visualize()

    def save_figs(self, path: str):
        self.plotter.save_figs(path)
