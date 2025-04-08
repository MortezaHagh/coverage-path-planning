""" Trapezoidal decomposition """
import math
from shapely import Polygon
import matplotlib.pyplot as plt
from scripts.decomposition.events import Event
from scripts.decomposition.obstacles import Obstacle


class Cell:

    def __init__(self, ind: int, poly: Polygon):
        self.ind = ind
        self.poly: Polygon = poly
        self.up_adj_ci = None
        self.down_adj_ci = None
        self.n_adj = 0

    def set_up_adj_ci(self, ci: str):
        self.up_adj_ci = ci
        self.n_adj += 1

    def set_down_adj_ci(self, ci: str):
        self.down_adj_ci = ci
        self.n_adj += 1


class TrapezoidalDecomposition:
    def __init__(self):
        self.ns = "TrapezoidalDecomposition"
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
        self.fig = None
        self.ax = None

    def initiate(self):
        self.create_obstacles()
        self.create_events()
        #
        self.decompose_all()
        # self.create_final_trapezoids()
        self.find_adjacents_cells()
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

    def add_cell(self, poly: Polygon):
        self.trapezoids.append(poly)
        cell = Cell(self.n_cells, poly)
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
            self.create_trapezoid(base_e, hit_up_event, hit_down_event, hit_up_y, hit_down_y)

            # update done status of lower events at same x coordinate
            if hit_up_event and i > 2:
                j = i-1
                lower_e = self.events[j]
                while lower_e.xy[0] == base_e.xy[0]:
                    j -= 1
                    lower_e = self.events[j]

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

            if base_e.can_up and base_e.etype in ('ceil', 'updown'):
                success_up, y_up = self.evel_event_up(base_e, e, hit_up_y)
            if base_e.can_down and base_e.etype in ('floor', 'updown'):
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
                if hit_up_event.etype == 'floor':
                    hit_up_event.done = True
                elif hit_up_event.etype == 'updown':
                    hit_up_event.can_down = False
        if hit_down_event:
            if hit_down_event.xy[0] == base_e.xy[0]:
                if hit_down_event.etype == 'ceil':
                    hit_down_event.done = True

    def create_trapezoid(self, base_e: Event, hit_up_event: Event, hit_down_event: Event, hit_up_y: float, hit_down_y: float):
        #
        if base_e.direct_trap:
            trap1 = [base_e.xy, base_e.end_down_ed_xy, base_e.end_up_ed_xy]
            base_e.add_endpoint_ci(self.n_cells)
            self.add_cell(Polygon(trap1))
            return

        #
        if not (hit_up_event or hit_down_event):
            # print((f"[{self.ns}] [WARN] create_trapezoid: hit_up_event and hit_down_event are both None"))
            # print("base_e:", base_e.sd_ind, base_e.etype, base_e.etype_2)
            return

        #
        trap_up, trap_down = None, None

        #
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

        #
        elif base_e.etype == 'ceil' and hit_up_event:
            trap_up = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.end_down_ed_xy, base_e.end_up_ed_xy]
        #
        elif base_e.etype == 'floor' and hit_down_event:
            trap_down = [base_e.xy, base_e.end_down_ed_xy, hit_down_event.end_up_ed_xy, (base_e.xy[0], hit_down_y)]

        #
        if hit_up_event:
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
            hit_up_event.add_endpoint_ci(self.n_cells)
            base_e.add_endpoint_ci(self.n_cells)
        if hit_down_event:
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))
            dn = 0
            if base_e.etype_2 == 'out':
                dn = 1
            hit_down_event.add_endpoint_ci(self.n_cells+dn)
            base_e.add_endpoint_ci(self.n_cells+dn)

        #
        if trap_up:
            self.add_cell(Polygon(trap_up))
        if trap_down:
            self.add_cell(Polygon(trap_down))
        if trap_up is None and trap_down is None:
            print((f"[{self.ns}][create_trapezoid][ERROR]: Invalid trapezoid creation"))

    def evel_event_up(self, base_e: Event, e: Event, y_hit: float) -> bool:
        # based even type is 'ceil'
        if e.etype == 'ceil':
            return False, None

        # check not to hit from outside of polygon
        if e.direct_trap and base_e.xy[0] > e.x_max:
            return False, None

        # reach to higher updown event
        new_y_hit = e.xy[1]
        if new_y_hit > base_e.y_max_lim:
            return False, None
        if e.xy[0] == base_e.xy[0] and new_y_hit < y_hit:
            if e.etype == 'updown':
                if e.etype_2 == 'out':
                    base_e.y_max_lim = y_hit
                    return True, new_y_hit
                return False, None

        # type is 'floor'
        # edge = e1-e
        e1_ind = e.prev_ind
        e1 = self.obstacles[e.obst_ind].events[e1_ind]

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
        # check y coordinates relative to max limit
        if new_y_hit > base_e.y_max_lim:
            return False, None
        if new_y_hit < y_hit:
            return True, new_y_hit
        #
        return False, None

    def eval_event_down(self, base_e: Event, e: Event, y_hit: float) -> bool:
        # based even type is 'floor'
        if e.etype == 'floor':
            return False, None

        # type is 'ceil'
        # edge = e1-e
        e2_ind = e.next_ind
        e2 = self.obstacles[e.obst_ind].events[e2_ind]

        # check coordinates
        # if e2 is the base event itself
        if e2.sd_ind == base_e.sd_ind:
            return False, None
        # if x base_e is left to e2
        if e2.xy[0] > base_e.xy[0]:
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

    def find_adjacents_cells(self):
        for e in self.events:
            ci_upe, ci_downe = None, None
            current_cell: Cell = None

            # hit_up_event
            if e.hit_up_event:
                hit_up_event = e.hit_up_event
                ci_curr = e.endpoint_cil.pop(0)
                ci_upe = hit_up_event.endpoint_cil.pop(0)
                if not ci_curr == ci_upe:
                    raise ValueError(f"[{self.ns}][find_adjacents_cells][ERROR]: ci_up and ci_upe are not equal")
                current_cell: Cell = self.cells[ci_curr]
                len_up_cil = len(hit_up_event.endpoint_cil)
                if len_up_cil > 0:
                    up_adj_ci = hit_up_event.endpoint_cil[0]
                    current_cell.set_up_adj_ci(up_adj_ci)
                else:
                    if hit_up_event.etype == 'floor':
                        obst_ind = hit_up_event.obst_ind
                        next_e = self.obstacles[obst_ind].events[hit_up_event.next_ind]
                        if len(next_e.endpoint_cil) > 0:
                            up_adj_ci = next_e.endpoint_cil[0]
                            current_cell.set_up_adj_ci(up_adj_ci)
                    else:
                        print(f"[{self.ns}][find_adjacents_cells][ERROR]: hit_up_event has no cil")
                        print(f"e: {e.ind}")
                        print(f"hit_up_event: {hit_up_event.ind}")

            # hit_down_event
            if e.hit_down_event:
                ci_curr = e.endpoint_cil.pop(0)
                ci_downe = e.hit_down_event.endpoint_cil.pop(0)
                if not ci_curr == ci_downe:
                    raise ValueError(f"[{self.ns}][find_adjacents_cells][ERROR]: ci_down and ci_downe are not equal")
                current_cell: Cell = self.cells[ci_curr]
                if len(e.hit_down_event.endpoint_cil) > 0:
                    down_adj_ci = e.hit_down_event.endpoint_cil[0]
                    current_cell.set_down_adj_ci(down_adj_ci)

            #
            if ci_downe and ci_upe:
                if not ci_downe == ci_upe:
                    if not e.etype_2 == 'out':
                        print(f"[{self.ns}][find_adjacents_cells][ERROR]: ci_down and ci_up are not equal")
                        print(f"ci_down: {ci_downe}, ci_up: {ci_upe}")
                        # raise ValueError(f"[{self.ns}][find_adjacents_cells][ERROR]: ci_up and ci_down are not equal")

            #
            if (e.ind < self.n_events-4) and current_cell and e.has_adj:
                if current_cell.n_adj == 0:
                    print(f"[{self.ns}][find_adjacents_cells][ERROR]: ci_up and ci_down are None")
                    print(f"e: {e.sd_ind}")
                    print(f"current_cell: {current_cell.ind}")
                    # raise ValueError(f"[{self.ns}][find_adjacents_cells][ERROR]: ci_up and ci_down are None")

    def create_final_trapezoids(self):
        if len(self.events) < 6:
            return
        if self.events[-2].fake:
            base_e = self.events[-2]
            base_e.etype_2 = 'in'
            hit_up_event = self.events[-1]
            hit_down_event = self.events[-3]
            hit_up_y = hit_up_event.xy[1]
            hit_down_y = hit_down_event.xy[1]
            self.create_trapezoid(base_e, hit_up_event, hit_down_event, hit_up_y, hit_down_y)

#  visualization ======================================================================================================

    def visualize_boarder_events(self):
        # plot boarder polygon
        x, y = self.b_poly.exterior.xy
        plt.plot(x, y, color='black', linewidth=2)
        #
        for e in self.b_obstacle.events:
            text1 = f"{e.etype}"
            text2 = f"{e.s_ind}-{e.ind}--[{e.prev_ind},{e.next_ind}]"
            plt.text(e.xy[0], e.xy[1], text1, fontsize=12, color='blue')
            plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

    def visualize_events(self):
        # n = 61
        # self.ax.plot(self.events[n].xy[0], self.events[n].xy[1], 'o', color='black', markersize=5)
        # display events
        for e in self.events:
            # ct = "concave" if e.concave else "convex"
            # text1 = f"{e.sd_ind}-[t1:{e.etype}]-[t2:{e.etype_2}]--{ct}"
            # #
            # text2 = f"{e.ind}--[{e.prev_ind},{e.next_ind}]"
            # plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

            # # name
            # text1 = e.name
            # self.ax.text(e.xy[0], e.xy[1], text1, fontsize=8, color='blue')

            # # endpoint_cil
            # txt = f"cil:{e.endpoint_cil}"
            # plt.text(e.xy[0], e.xy[1]-3, txt, fontsize=10, color='blue')

            #
            # plt.plot(e.xy[0], e.xy[1], 'o', color='purple', markersize=8)
            if e.hit_up_event is not None:
                # plt.plot(e.xy[0], e.hit_up_y, 'o', color='black', markersize=5)
                self.ax.plot([e.xy[0], e.xy[0]], [e.xy[1], e.hit_up_y], '--', color='orange', linewidth=1)
            if e.hit_down_event is not None:
                # plt.plot(e.xy[0], e.hit_down_y, 'o', color='black', markersize=5)
                self.ax.plot([e.xy[0], e.xy[0]], [e.xy[1], e.hit_down_y], '--', color='orange', linewidth=1)

    def vizualize_cells(self):
        for cell in self.cells:
            # fill
            x, y = cell.poly.exterior.xy
            self.ax.fill(x, y, alpha=0.5, fc='yellow', ec='black', linewidth=1)
            # centroid
            self.ax.plot(cell.poly.centroid.x, cell.poly.centroid.y, 'o', color='purple', markersize=8)

            # # text
            # plt.text(cell.poly.centroid.x, cell.poly.centroid.y, f"[{cell.ind}]", fontsize=5, color='black')
            # self.ax.text(cell.poly.centroid.x, cell.poly.centroid.y,
            #              f"[{cell.ind}][{cell.down_adj_ci}, {cell.up_adj_ci}]", fontsize=5, color='black')

            # adjacents
            if cell.up_adj_ci:
                self.ax.plot([cell.poly.centroid.x, self.cells[cell.up_adj_ci].poly.centroid.x],
                             [cell.poly.centroid.y, self.cells[cell.up_adj_ci].poly.centroid.y], 'r--', linewidth=1)
            if cell.down_adj_ci:
                self.ax.plot([cell.poly.centroid.x, self.cells[cell.down_adj_ci].poly.centroid.x],
                             [cell.poly.centroid.y, self.cells[cell.down_adj_ci].poly.centroid.y], 'g--', linewidth=1)

    def visualize(self):
        # plot boarder
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        self.ax = ax

        # plot obstacles
        for obs in self.obstacles:
            x, y = obs.poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=1)

        # vizualize events
        self.visualize_events()

        # plot trapezoids
        self.vizualize_cells()

        #
        self.fig = fig
        plt.show()
