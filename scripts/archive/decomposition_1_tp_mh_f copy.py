""" Trapezoidal decomposition """
import math
from shapely import Polygon
from scripts.decomposition.events import Event
from scripts.decomposition.obstacles import Obstacle
from scripts.decomposition.cell import Cell, CellGroup
from scripts.decomposition.plotter import DecPlotter


class MHDecomposition:
    def __init__(self):
        self.ns = "MHDecomposition"
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
        self.deadend_cells: list[Cell] = []
        #
        self.plotter: DecPlotter = None

    def initiate(self):
        self.create_obstacles()
        self.create_events()
        #
        self.decompose_all()
        # self.create_final_trapezoids()
        self.find_adjacents_cells()
        self.find_adjacent_for_deadend_cells_2()
        self.create_cell_groups()
        self.visualize()

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

            # Set trapezoid events cell indices
            self.set_trap_events_cell_inds(base_e, hit_up_event, hit_down_event)

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

    def eval_done_status(self, base_e: Event, hit_up_event: Event, hit_down_event: Event):
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

    def set_trap_events_cell_inds(self, base_e: Event, hit_up_event: Event, hit_down_event: Event):
        # update done status of lower events at same x coordinate # out
        #
        if base_e.sd_ind in [136, 137, 144]:
            pass

        # direct_trap
        if base_e.direct_trap:
            next_e = base_e.next_e
            prev_e = base_e.prev_e
            if prev_e.xy[0] > next_e.xy[0]:
                self.cells[base_e.base_ci_up].add_event(prev_e)
            elif prev_e.xy[0] < next_e.xy[0]:
                self.cells[base_e.base_ci_up].add_event(next_e)
            else:
                self.cells[base_e.base_ci_up].add_event(prev_e)
                self.cells[base_e.base_ci_up].add_event(next_e)

        if hit_up_event:
            up_ind = base_e.sd_ind+1
            up_e = self.events[up_ind]
            while up_e.xy[0] == base_e.xy[0] and up_e.xy[1] < base_e.hit_up_y:
                up_e.done = True
                self.cells[base_e.base_ci_up].add_event(up_e)
                up_ind = up_ind + 1
                up_e = self.events[up_ind]

        #
        if hit_down_event:
            if hit_down_event.xy[0] == base_e.xy[0]:
                self.cells[base_e.base_ci_down].add_event(hit_down_event)
            #
            e = self.get_event(base_e.sd_ind-1)
            if e.fake:
                self.cells[base_e.base_ci_down].add_event(e)
                e = self.get_event(base_e.sd_ind-2)
                self.cells[base_e.base_ci_down].add_event(e)
            #
            if len(hit_down_event.up_cil_k) == 1:
                self.cells[base_e.base_ci_down].add_event(hit_down_event.next_e)

        #
        if hit_up_event:
            if hit_up_event.xy[0] == base_e.xy[0]:
                self.cells[base_e.base_ci_up].add_event(hit_up_event)
            #
            e = self.get_event(base_e.sd_ind+1)
            if e.fake:
                self.cells[base_e.base_ci_up].add_event(e)
                e = self.get_event(base_e.sd_ind+2)
                self.cells[base_e.base_ci_up].add_event(e)
            #
            if len(hit_up_event.down_cil_k) == 1:
                self.cells[base_e.base_ci_up].add_event(hit_up_event.prev_e)
                #
                low_j = hit_up_event.prev_e.sd_ind-1
                low_e = self.events[low_j]
                if base_e.end_up_ed_xy:
                    while low_e.xy[0] == hit_up_event.prev_e.xy[0] and low_e.xy[1] > base_e.end_up_ed_xy[1]:
                        self.cells[base_e.base_ci_up].add_event(low_e)
                        low_j = low_j-1
                        low_e = self.events[low_j]
                elif not base_e.fake:
                    print("no end_up_ed_xy", base_e.sd_ind)

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

    # find_adjacents_cells =============================================================================================

    def find_adjacents_cells(self):
        for e in self.events:
            up_adj_ci, down_adj_ci = None, None
            current_cell: Cell = None

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

            # end if hit_up_event:

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

    def find_adjacent_for_deadend_cells(self):
        for cell in self.cells[::-1]:
            e = cell.event

            # case n_adj = 0 --------------------------------------------------
            if e.has_adj and cell.n_adj == 0:
                if e.etype_2 == 'out' or e.etype == 'up':
                    up_e = e.hit_up_event
                    while up_e and e.xy[0] == up_e.xy[0]:
                        up_cell_i = up_e.base_ci_up
                        if up_cell_i:
                            up_cell = self.cells[up_cell_i]
                            if up_cell.up_adj_ci:
                                cell.set_up_adj_ci(up_cell.up_adj_ci)
                                self.cells[up_cell.up_adj_ci].set_parent_up_ci(cell.ind)
                                break
                        n_up_e = up_e.hit_up_event
                        if n_up_e:
                            up_e = n_up_e
                        else:
                            if up_e.vertical:
                                up_e = up_e.next_e
                                if up_e.fake:
                                    up_e = up_e.next_e
                            else:
                                break
                #
                if cell.n_adj == 0:
                    cell.status = False
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: No adjacents found:")
                    print(f"e: {e.sd_ind}, -- cell: {cell.ind}")

            # case n_parent = 0 ------------------------------------------------
            if e.has_parent and cell.n_parent == 0:
                # case: upper cell has parent which is also adjacent of current cell
                # b_e_ind = None
                b_e: Event = None
                if e.etype == 'down' or e.etype_2 == 'out':
                    b_e = e.prev_e
                elif e.etype == 'up':
                    b_e = e.next_e
                elif e.etype_2 == 'in':
                    up_e = e.hit_up_event
                    if up_e:
                        b_e = up_e.prev_e
                if b_e:
                    down_e = self.events[b_e.sd_ind]
                    while down_e.done and down_e.xy[0] == b_e.xy[0]:
                        down_e = self.events[down_e.sd_ind-1]
                    if down_e and down_e.base_ci_down:
                        cell.add_mid_parentlist(down_e.base_ci_down)
                        self.cells[down_e.base_ci_down].add_mid_adjlist(cell.ind)
                        continue
                    elif down_e and down_e.base_ci_up:
                        cell.add_mid_parentlist(down_e.base_ci_up)
                        self.cells[down_e.base_ci_up].add_mid_adjlist(cell.ind)
                        continue
                #
                if e.direct_trap:
                    ee = e.next_e
                    if e.prev_e.xy[0] > e.next_e.xy[0]:
                        ee = e.prev_e
                    down_e = ee
                    while down_e.done and ee.xy[0] == down_e.xy[0]:
                        down_e = self.events[down_e.sd_ind-1]
                    if down_e.base_ci_up:
                        cell.add_mid_parentlist(down_e.base_ci_up)
                        self.cells[down_e.base_ci_up].add_mid_adjlist(cell.ind)

                #
                if cell.n_parent == 0:
                    cell.status = False
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: No parents found:")
                    print(f"e: {cell.event.sd_ind}, -- cell: {cell.ind}")

    def find_adjacent_for_deadend_cells_2(self):
        for cell in self.cells[::-1]:
            e = cell.event
            c_events = cell.events

            # case n_adj = 0 --------------------------------------------------
            if e.has_adj and cell.n_adj == 0:
                # iterate over all events of the cell
                for ce in c_events:
                    # try finding next added cell to this event
                    if cell.ind in ce.cell_inds:
                        ind = ce.cell_inds.index(cell.ind)
                        if len(ce.cell_inds) > ind+1:
                            a_cind = ce.cell_inds[ind+1]
                            adj_cell = self.cells[a_cind]
                            if ce.xy[0] < adj_cell.xy[0]:
                                cell.add_mid_adjlist(a_cind)
                                adj_cell.add_mid_parentlist(cell.ind)
                                break
                    else:
                        raise ValueError(f"[{self.ns}][find_adjacent_for_deadend_cells_2] case1")

                if cell.n_adj == 0:
                    cell.status = False
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: No adjacents found:")
                    print(f"e: {e.sd_ind}, -- cell: {cell.ind}")

            # case n_parent = 0 ------------------------------------------------
            if e.has_parent and cell.n_parent == 0:
                # iterate over all events of the cell
                for ce in c_events:
                    # try finding next added cell to this event
                    if cell.ind in ce.cell_inds:
                        ind = ce.cell_inds.index(cell.ind)
                        if ind > 0:
                            a_cind = ce.cell_inds[ind-1]
                            cell.add_mid_parentlist(a_cind)
                            self.cells[a_cind].add_mid_adjlist(cell.ind)
                            break
                    else:
                        print("cell.ind", cell.ind)
                        print("event", ce.sd_ind)
                        raise ValueError(f"[{self.ns}][find_adjacent_for_deadend_cells_2] case2")

                #
                if cell.n_parent == 0:
                    cell.status = False
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: No parents found:")
                    print(f"e: {cell.event.sd_ind}, -- cell: {cell.ind}")

    def create_cell_groups(self):
        cell_groups = []
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
                cell_groups.append(new_group)
                ncg += 1

        #
        self.cells_groups = cell_groups

    def get_event(self, ind: int) -> Event:
        if ind < 0 or ind >= len(self.events):
            return None
        return self.events[ind]

    def visualize(self):
        self.plotter = DecPlotter(self)
        self.plotter.visualize()

    def save_figs(self, path: str):
        self.plotter.save_figs(path)
