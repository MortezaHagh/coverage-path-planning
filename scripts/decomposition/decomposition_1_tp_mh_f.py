""" Trapezoidal decomposition """
import math
import matplotlib.pyplot as plt
from shapely import Polygon, Point


class Event:
    def __init__(self, obst_ind, ind, v):
        self.xy = v
        self.x_max = None
        #
        self.etype: str = None      # up, down, updown
        self.etype_2: str = None    # out, in
        self.etype_3: str = None    # ovex, ivex, ocave, icave
        self.c_etype: str = None    # up, down, updown
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
        self.hitted_event: Event = None
        self.endp_up_e: tuple[float, float] = None
        self.endp_down_e: tuple[float, float] = None
        #
        self.fake: bool = False
        self.vertical: bool = False
        #
        self.concave = False
        self.convex = True
        self.skip_sweep = False
        self.direct_trap = False
        #
        self.done = False
        self.can_up = True
        self.can_down = True
        #

    def set_s_ind(self, ind: int):
        self.s_ind = ind

    def set_sd_ind(self, ind: int):
        self.sd_ind = ind

    def set_prev_ind(self, ind: int):
        self.prev_ind = ind

    def set_next_ind(self, ind: int):
        self.next_ind = ind

    def set_endpoint_up_edge(self, xy: tuple[float, float]):
        self.endp_up_e = xy

    def set_endpoint_down_edge(self, xy: tuple[float, float]):
        self.endp_down_e = xy

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
        self.name = name

# Obstacle ====================================================================


class Obstacle:
    def __init__(self, ind: int, poly: Polygon):
        self.ind: int = ind
        self.poly: Polygon = poly
        self.events: list[Event] = []
        self.s_events: list[Event] = []
        self.n_v: int = len(self.poly.exterior.coords)-1
        #
        self.create_events()
        self.create_fake_events()
        self.create_sorted_events()
        self.eval_concave()
        if self.ind == 0:
            self.process_events_bounding()
        else:
            self.process_events_types_1()
        self.process_fake_events()
        # self.print_events()

    def create_events(self):
        vertices = self.poly.exterior.coords[:-1]
        self.events = [Event(self.ind, i, v) for i, v in enumerate(vertices)]

    def create_fake_events(self):
        n_fake = 0
        temp_dict = {}
        for i, e in enumerate(self.events[:self.n_v]):
            j = i+1
            if j == self.n_v:
                n_fake = 0
                j = 0
            if e.xy[0] == self.events[j].xy[0]:
                self.events[i].vertical = True
                self.events[j].vertical = True
                e1 = self.events[i]
                e2 = self.events[j]
                xy = (e.xy[0], (e1.xy[1] + e2.xy[1]) / 2)
                # create fake event
                fake_event = Event(self.ind, j+n_fake, xy)
                fake_event.etype = 'updown'
                fake_event.c_etype = 'updown'
                fake_event.fake = True
                temp_dict[j+n_fake] = fake_event
                n_fake += 1
                for ee in self.events[j:]:
                    ee.ind += 1
        # add fake events to the events list
        for k, v in temp_dict.items():
            self.events.insert(k, v)
        # sort based on ind
        self.n_v = len(self.events)
        self.events = sorted(self.events, key=lambda e: (e.ind))

    def create_sorted_events(self):
        self.s_events = sorted(self.events, key=lambda e: (e.xy[0], e.xy[1]))
        for i, e in enumerate(self.s_events):
            e.set_s_ind(i)
            prev_ind = (self.s_events[i].ind - 1) % (self.n_v)
            next_ind = (self.s_events[i].ind + 1) % (self.n_v)
            e.set_prev_ind(prev_ind)
            e.set_next_ind(next_ind)

    def eval_concave(self) -> bool:
        for e in self.events:
            x, y = e.xy
            prev_x, prev_y = self.events[e.prev_ind].xy
            next_x, next_y = self.events[e.next_ind].xy
            # Check if the event is concave
            e.concave = (prev_x - x) * (next_y - y) - (prev_y - y) * (next_x - x) > 0
            e.convex = not e.concave

    def process_events_bounding(self):
        for e in self.events:

            # swap prev and next indices
            e.prev_ind, e.next_ind = e.next_ind, e.prev_ind

            # x, y, preve, nexte
            x, y = e.xy
            next_e = self.events[e.next_ind]
            prev_e = self.events[e.prev_ind]

            # set in-out for fake events
            step = 1.0
            if e.fake:
                # e.etype_2 = 'in'
                if self.poly.contains(Point(x+step, y)):
                    e.etype_2 = 'in'
                elif self.poly.contains(Point(x-step, y)):
                    e.etype_2 = 'out'
                else:
                    raise ValueError("Invalid event type [fake]")
                #
                if e.etype_2 == 'in' and e.convex:
                    e.etype_3 = "ivex"
                    e.skip_sweep = True
                continue

            # detect event types: up, down, updown, in, out
            if x < prev_e.xy[0] and x < next_e.xy[0]:
                e.etype = 'updown'
                e.etype_2 = 'in'
            elif x > prev_e.xy[0] and x > next_e.xy[0]:
                e.etype = 'updown'
                e.etype_2 = 'out'
            elif prev_e.xy[0] < x < next_e.xy[0]:
                e.etype = 'down'
            elif next_e.xy[0] < x < prev_e.xy[0]:
                e.etype = 'up'
            elif x == prev_e.xy[0] or x == next_e.xy[0]:
                c_up = self.poly.contains(Point(x, y + step)) or self.poly.touches(Point(x, y + step))
                c_down = self.poly.contains(Point(x, y - step)) or self.poly.touches(Point(x, y - step))
                if c_up and c_down:
                    e.etype = 'updown'
                    if self.poly.contains(Point(x+step, y)):
                        e.etype_2 = 'out'
                    elif self.poly.contains(Point(x-step, y)):
                        e.etype_2 = 'in'
                    else:
                        raise ValueError("Invalid event type [in-out-2]")
                elif c_up:
                    e.etype = 'up'
                elif c_down:
                    e.etype = 'down'
                else:
                    raise ValueError("Invalid event type [up-down]")
            else:
                raise ValueError("Invalid event type [in-out]")

            # set endpoints
            if e.etype == 'updown':
                e.c_etype = 'updown'
                e.set_endpoint_down_edge(prev_e.xy)
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'up':
                e.c_etype = 'down'
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'down':
                e.c_etype = 'up'
                e.set_endpoint_down_edge(prev_e.xy)
            else:
                raise ValueError("Invalid event type [e.etype]")

            # detect skipped sweep events
            if e.etype_2 == 'in':
                if e.convex:
                    e.etype_3 = "ivex"
                    e.skip_sweep = True
            elif e.etype_2 == 'out' and e.convex:
                e.etype_3 = "ovex"
                e.direct_trap = True
                e.x_max = max(prev_e.xy[0], next_e.xy[0])

            #
            if e.vertical:
                e.skip_sweep = True

    def process_events_types_1(self):
        for e in self.events:
            # x, y, preve, nexte
            x, y = e.xy
            next_e = self.events[e.next_ind]
            prev_e = self.events[e.prev_ind]

            # set in-out for fake events
            if e.fake:
                step = 1.0
                if self.poly.contains(Point(x+step, y)):
                    e.etype_2 = 'in'
                elif self.poly.contains(Point(x-step, y)):
                    e.etype_2 = 'out'
                else:
                    raise ValueError("Invalid event type [fake]")
                #
                if e.etype_2 == 'in' and e.concave:
                    e.etype_3 = "icave"
                    e.skip_sweep = True
                continue

            # detect event types: up, down, updown, in, out
            if x < prev_e.xy[0] and x < next_e.xy[0]:
                e.etype = 'updown'
                e.etype_2 = 'in'
            elif x > prev_e.xy[0] and x > next_e.xy[0]:
                e.etype = 'updown'
                e.etype_2 = 'out'
            elif prev_e.xy[0] < x < next_e.xy[0]:
                e.etype = 'down'
            elif next_e.xy[0] < x < prev_e.xy[0]:
                e.etype = 'up'
            elif x == prev_e.xy[0] or x == next_e.xy[0]:
                step = 1.0
                if self.poly.touches(Point(x, y + step)):
                    e.etype = 'up'
                elif self.poly.touches(Point(x, y - step)):
                    e.etype = 'down'
                else:
                    raise ValueError("Invalid event type [up-down]")
            else:
                raise ValueError("Invalid event type [in-out]")

            # set endpoints
            if e.etype == 'updown':
                e.c_etype = 'updown'
                e.set_endpoint_down_edge(prev_e.xy)
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'up':
                e.c_etype = 'down'
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'down':
                e.c_etype = 'up'
                e.set_endpoint_down_edge(prev_e.xy)
            else:
                raise ValueError("Invalid event type [e.etype]")

            # detect skipped sweep events
            if e.etype_2 == 'in':
                if e.concave:
                    e.etype_3 = "icave"
                    e.skip_sweep = True
            elif e.etype_2 == 'out' and e.concave:
                e.etype_3 = "ocave"
                e.direct_trap = True
                e.x_max = max(prev_e.xy[0], next_e.xy[0])

            #
            if e.vertical:
                e.skip_sweep = True

    def process_fake_events(self):
        for e in self.events:
            if not e.fake:
                continue
            prev_e = self.events[e.prev_ind]
            next_e = self.events[e.next_ind]
            #
            if (prev_e.etype in ['up', 'updown']) and (next_e.etype in ['up', 'updown']):
                e.skip_sweep = True
                prev_e.skip_sweep = True
                # prev_e.vertical = True
                next_e.skip_sweep = False
            elif (prev_e.etype in ['down', 'updown']) and (next_e.etype in ['down', 'updown']):
                e.skip_sweep = True
                prev_e.skip_sweep = True
                # next_e.vertical = True
                next_e.skip_sweep = False

    def inverse_event_types(self):
        for e in self.events:
            temp = e.prev_ind
            e.prev_ind = e.next_ind
            e.next_ind = temp
            # swpap endpoints
            e.endp_up_e, e.endp_down_e = e.endp_down_e, e.endp_up_e

    def print_events(self):
        for e in self.events:
            print(f"Event {e.ind}: {e.xy} {e.etype}")

# TrapezoidalDecomposition =============================================================================================


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
        self.create_final_trapezoids()
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

            if base_e.can_up and base_e.etype in ('up', 'updown'):
                success_up, y_up = self.evel_event_up(base_e, e, hit_up_y)
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
        if hit_up_event is not None:
            if hit_up_event.xy[0] == base_e.xy[0]:
                if hit_up_event.etype == 'down':
                    hit_up_event.done = True
        if hit_down_event is not None:
            if hit_down_event.xy[0] == base_e.xy[0]:
                if hit_down_event.etype == 'up':
                    hit_down_event.done = True

    def create_trapezoid(self, base_e: Event, hit_up_event: Event, hit_down_event: Event, hit_up_y: float, hit_down_y: float):
        #
        if base_e.direct_trap:
            trap1 = [base_e.xy, base_e.endp_down_e, base_e.endp_up_e]
            self.trapezoids.append(Polygon(trap1))
            return

        #
        if not (hit_up_event or hit_down_event):
            print((f"[{self.ns}] [WARN] create_trapezoid: hit_up_event and hit_down_event are both None"))
            print("base_e:", base_e.sd_ind, base_e.etype, base_e.etype_2)
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
                if hit_down_event:
                    trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e,
                             hit_down_event.endp_up_e, (base_e.xy[0], hit_down_y)]
            if not (hit_up_event or hit_down_event):
                print((f"[{self.ns}] [WARN] updown: hit_up_event and hit_down_event are None"))
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))
        #
        elif base_e.etype == 'up' and hit_up_event:
            trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e, base_e.endp_up_e]
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
        #
        elif base_e.etype == 'down' and hit_down_event:
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
        # based even type is "up"
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
                    e.can_down = False
                    base_e.can_up = False
                    return True, new_y_hit
                e.done = True
                return False, None

        # type is "down"
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
        # based even type is "down"
        if e.etype == 'down':
            return False, None

        # type is "up"
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_aspect('equal', adjustable='box')

        # plot obstacles
        for obs in self.obstacles:
            x, y = obs.poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=1)

        # display events
        for e in self.events:
            # ct = "concave" if e.concave else "convex"
            # text1 = f"{e.sd_ind}-[t1:{e.etype}]-[t2:{e.etype_2}]--{ct}"
            text1 = e.name
            plt.text(e.xy[0], e.xy[1], text1, fontsize=10, color='blue')
            #
            # text2 = f"{e.s_ind}-{e.ind}--[{e.prev_ind},{e.next_ind}]"
            # plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

            #
            plt.plot(e.xy[0], e.xy[1], 'o', color='purple', markersize=8)
            if e.hit_up_event is not None:
                plt.plot(e.xy[0], e.hit_up_y, 'o', color='black', markersize=5)
                plt.plot([e.xy[0], e.xy[0]], [e.xy[1], e.hit_up_y], '--', color='orange', linewidth=1)
            if e.hit_down_event is not None:
                plt.plot(e.xy[0], e.hit_down_y, 'o', color='black', markersize=5)
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
    p2 = Polygon([(30, 30), (40, 20), (60, 30), (70, 40), (60, 50), (30, 50), (20, 40)])
    return [p1, p2]


if __name__ == "__main__":
    td = TrapezoidalDecomposition()
    td.set_bounding_polygon(create_bounding())
    td.set_obstacle_polygons(create_polygons())
    td.initiate()
