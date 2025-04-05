""" Trapezoidal decomposition """
import math
from shapely import Polygon
import matplotlib.pyplot as plt
from scripts.decomposition.events import Event
from scripts.decomposition.obstacles import Obstacle


class TrapezoidalDecomposition:
    def __init__(self):
        self.ns = "TrapezoidalDecomposition"
        self.b_poly: Polygon = None
        self.b_obstacle: Obstacle = None
        self.polygons: list[Polygon] = []
        self.obstacles: list[Obstacle] = []
        self.events: list[Event] = []
        self.trapezoids: list[Polygon] = []
        self.n_events: int = 0
        #
        self.fig = None

    def initiate(self):
        self.create_obstacles()
        self.create_events()
        #
        self.decompose_all()
        # self.create_final_trapezoids()
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

    # decompose_all ============================================================
    def decompose_all(self):
        for i, base_e in enumerate(self.events[1:]):
            ind = i + 1

            if base_e.sd_ind == 421:
                pass

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

    def expand_hit(self, base_e: Event, base_i: int):
        # Initialize hit variables
        hit_up_y, hit_down_y = math.inf, -math.inf
        hit_up_event, hit_down_event = None, None

        # check if the event needs direct trapezoid
        if base_e.direct_trap:
            return hit_up_y, hit_down_y, hit_up_event, hit_down_event

        # Iterate over events
        for e in self.events[base_i+1:]:
            if e.fake:
                continue

            # check fake vertices
            if base_e.fake and base_e.obst_ind == e.obst_ind:
                continue

            # if (base_e.fake and e.vertical) and base_e.obst_ind == e.obst_ind:  # and (abs(base_e.sd_ind - e.sd_ind) == 1):
            #     continue

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
        if hit_down_event:
            if hit_down_event.xy[0] == base_e.xy[0]:
                if hit_down_event.etype == 'ceil':
                    hit_down_event.done = True

    def create_trapezoid(self, base_e: Event, hit_up_event: Event, hit_down_event: Event, hit_up_y: float, hit_down_y: float):
        #
        if base_e.direct_trap:
            trap1 = [base_e.xy, base_e.endp_down_e, base_e.endp_up_e]
            self.trapezoids.append(Polygon(trap1))
            return

        #
        if not (hit_up_event or hit_down_event):
            # print((f"[{self.ns}] [WARN] create_trapezoid: hit_up_event and hit_down_event are both None"))
            # print("base_e:", base_e.sd_ind, base_e.etype, base_e.etype_2)
            return

        #
        trap1, trap2 = None, None

        #
        if base_e.etype == 'updown':
            if base_e.etype_2 == 'out':
                if hit_up_event:
                    trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e, base_e.endp_up_e]
                if hit_down_event:
                    trap2 = [base_e.xy, (base_e.xy[0], hit_down_y), hit_down_event.endp_up_e, base_e.endp_down_e]
            else:
                if hit_down_event and hit_up_event:
                    trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e,
                             hit_down_event.endp_up_e, (base_e.xy[0], hit_down_y)]
                else:
                    print((f"[{self.ns}] [WARN] create_trapezoid: hit_up_event and hit_down_event are both None [updown-in]"))
                    return

            if not (hit_up_event or hit_down_event):
                print((f"[{self.ns}] [WARN] updown: hit_up_event and hit_down_event are None"))
            if hit_up_event:
                hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
            if hit_down_event:
                hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))
        #
        elif base_e.etype == 'ceil' and hit_up_event:
            trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e, base_e.endp_up_e]
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
        #
        elif base_e.etype == 'floor' and hit_down_event:
            trap1 = [base_e.xy, base_e.endp_down_e, hit_down_event.endp_up_e, (base_e.xy[0], hit_down_y)]
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))

        #
        if trap1:
            self.trapezoids.append(Polygon(trap1))
        if trap2:
            self.trapezoids.append(Polygon(trap2))
        if trap1 is None and trap2 is None:
            print((f"[{self.ns}], [ERROR] Invalid trapezoid creation"))

    def evel_event_up(self, base_e: Event, e: Event, y_hit: float) -> bool:
        # based even type is 'ceil'
        if e.etype == 'ceil':
            return False, None

        # check not to hit from outside of polygon
        if e.direct_trap and base_e.xy[0] > e.x_max:
            return False, None

        # reach to higher updown event # todo y_lim
        new_y_hit = e.xy[1]
        if e.xy[0] == base_e.xy[0] and new_y_hit < y_hit:
            if e.etype == 'updown':
                if e.etype_2 == 'out':
                    e.can_down = False
                    base_e.can_up = False
                    return True, new_y_hit
                e.done = True
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

    def visualize(self):
        # plot boarder
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')

        # plot obstacles
        for obs in self.obstacles:
            x, y = obs.poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=1)

        # display events
        for e in self.events:
            # ct = "concave" if e.concave else "convex"
            # text1 = f"{e.sd_ind}-[t1:{e.etype}]-[t2:{e.etype_2}]--{ct}"
            # #
            # text2 = f"{e.ind}--[{e.prev_ind},{e.next_ind}]"
            # plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

            #
            text1 = e.name
            plt.text(e.xy[0], e.xy[1], text1, fontsize=10, color='blue')

            #
            # plt.plot(e.xy[0], e.xy[1], 'o', color='purple', markersize=8)
            if e.hit_up_event is not None:
                # plt.plot(e.xy[0], e.hit_up_y, 'o', color='black', markersize=5)
                plt.plot([e.xy[0], e.xy[0]], [e.xy[1], e.hit_up_y], '--', color='orange', linewidth=1)
            if e.hit_down_event is not None:
                # plt.plot(e.xy[0], e.hit_down_y, 'o', color='black', markersize=5)
                plt.plot([e.xy[0], e.xy[0]], [e.xy[1], e.hit_down_y], '--', color='orange', linewidth=1)

        # plot trapezoids
        for i, trap in enumerate(self.trapezoids):
            x, y = trap.exterior.xy
            plt.fill(x, y, alpha=0.5, fc='yellow', ec='black', linewidth=1)
            plt.text(trap.centroid.x, trap.centroid.y, f"[{i}]", fontsize=8, color='black')

        #
        self.fig = fig
        plt.show()


# example
def create_bounding():
    # create boarder polygon
    b_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    return b_poly


def create_polygons():
    # create obstacles polygons
    p1 = Polygon([(40, 60), (70, 60), (80, 66), (70, 70), (50, 80), (40, 70)])
    p2 = Polygon([(30, 30), (40, 20), (60, 30), (60, 50), (30, 50), (20, 40)])
    # p2 = Polygon([(30, 30), (40, 20), (60, 30), (70, 40), (60, 50), (30, 50), (20, 40)])
    return [p1, p2]


if __name__ == "__main__":
    td = TrapezoidalDecomposition()
    td.set_bounding_polygon(create_bounding())
    td.set_obstacle_polygons(create_polygons())
    td.initiate()
