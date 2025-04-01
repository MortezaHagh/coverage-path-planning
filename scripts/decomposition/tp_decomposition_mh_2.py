""" Trapezoidal decomposition """
import math
from shapely import Polygon, Point
import matplotlib.pyplot as plt


class Event:
    def __init__(self, obst_ind, ind, v):
        self.etype: str = "updown"      # up, down, updown
        self.etype_2: str = "normal"    # normal, out, in
        self.c_etype: str = "updown"
        self.fake: bool = False
        self.vertical: bool = False
        self.obst_ind: int = obst_ind
        self.ind: int = ind         # index in the obstacle (ccw)
        self.s_ind: int = None      # sorted index in the obstacles events
        self.sd_ind: int = None     # sorted index in the decomposition
        self.prev_ind: int = None
        self.next_ind: int = None
        self.xy = v
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
        self.done = False

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
        if self.ind == 0:
            self.process_events_bounding()
        else:
            self.process_events_types_1()
        # self.print_events()

    def create_events(self):
        vertices = self.poly.exterior.coords[:-1]
        self.events = [Event(self.ind, i, v) for i, v in enumerate(vertices)]

    def create_sorted_events(self):
        self.s_events = sorted(self.events, key=lambda e: (e.xy[0], e.xy[1]))
        for i, e in enumerate(self.s_events):
            e.set_s_ind(i)
            prev_ind = (self.s_events[i].ind - 1) % (self.n_v)
            next_ind = (self.s_events[i].ind + 1) % (self.n_v)
            e.set_prev_ind(prev_ind)
            e.set_next_ind(next_ind)

    def process_events_types_1(self):
        for e in self.events:
            if e.fake:
                continue
            #
            x, y = e.xy
            step = 1.0  # Set step size as a parameter
            # Check if the point in +y direction is inside or on the boundary of the obstacle
            if self.poly.contains(Point(x, y + step)) or self.poly.touches(Point(x, y + step)):
                e.etype = 'down'
                e.c_etype = 'up'
                e.set_endpoint_down_edge(self.events[e.prev_ind].xy)
            # Check if the point in -y direction is inside or on the boundary of the obstacle
            elif self.poly.contains(Point(x, y - step)) or self.poly.touches(Point(x, y - step)):
                e.etype = 'up'
                e.c_etype = 'down'
                e.set_endpoint_up_edge(self.events[e.next_ind].xy)
            else:
                e.set_endpoint_up_edge(self.events[e.next_ind].xy)
                e.set_endpoint_down_edge(self.events[e.prev_ind].xy)
                # set type 2
                if not e.fake:
                    x = e.xy[0]
                    x_next = self.events[e.next_ind].xy[0]
                    x_prev = self.events[e.prev_ind].xy[0]
                    if x > x_next and x > x_prev:
                        e.etype_2 = "out"
                    elif x < x_next and x < x_prev:
                        e.etype_2 = "in"

    def process_events_types_2(self):
        for i, e1 in enumerate(self.events):
            for e2 in self.events[i+1:]:
                if e1.xy[0] == e2.xy[0]:
                    e1.etype, e2.etype = ('up', 'down') if (e1.xy[1] > e2.xy[1]) else ('down', 'up')

    def process_events_bounding(self):
        for e in self.events:
            if e.fake:
                continue
            #
            x, y = e.xy
            step = 1.0  # Set step size as a parameter
            case_1, case_2, case_3 = False, False, False
            # Check if the point in +y direction is inside or on the boundary of the obstacle
            if self.poly.contains(Point(x, y + step)) or self.poly.touches(Point(x, y + step)):
                case_1 = True
            # Check if the point in -y direction is inside or on the boundary of the obstacle
            if self.poly.contains(Point(x, y - step)) or self.poly.touches(Point(x, y - step)):
                case_2 = True
            if case_1 and case_2:
                case_3 = True

            if case_3:
                e.set_endpoint_up_edge(self.events[e.next_ind].xy)
                e.set_endpoint_down_edge(self.events[e.prev_ind].xy)
                # set type 2
                if not e.fake:
                    x = e.xy[0]
                    x_next = self.events[e.next_ind].xy[0]
                    x_prev = self.events[e.prev_ind].xy[0]
                    if x > x_next and x > x_prev:
                        e.etype_2 = "out"
                    elif x < x_next and x < x_prev:
                        e.etype_2 = "in"
            elif case_1:
                e.etype = 'up'
                e.c_etype = 'down'
                e.set_endpoint_down_edge(self.events[e.prev_ind].xy)
            elif case_2:
                e.etype = 'down'
                e.c_etype = 'up'
                e.set_endpoint_up_edge(self.events[e.next_ind].xy)

    def inverse_event_types(self):
        for e in self.events:
            temp = e.prev_ind
            e.prev_ind = e.next_ind
            e.next_ind = temp
            #
            # if e.etype == 'up':
            #     e.etype = 'down'
            # elif e.etype == 'down':
            #     e.etype = 'up'
            #
            # swpap endpoints
            e.endp_up_e, e.endp_down_e = e.endp_down_e, e.endp_up_e

    def print_events(self):
        for e in self.events:
            print(f"Event {e.ind}: {e.xy} {e.etype}")

    def create_fake_events(self):
        n_fake = 0
        for i, e in enumerate(self.events[:self.n_v]):
            j = i+1
            if j == self.n_v:
                j = 0
            if e.xy[0] == self.events[j].xy[0]:
                self.events[i].vertical = True
                self.events[j].vertical = True
                e1 = self.events[i]
                e2 = self.events[j]
                xy = (e.xy[0], (e1.xy[1] + e2.xy[1]) / 2)
                # create fake event
                fake_event = Event(self.ind, i+1+n_fake, xy)
                fake_event.etype = 'updown'
                fake_event.c_etype = 'updown'
                fake_event.fake = True
                n_fake += 1
                self.events.append(fake_event)
                if j > 0:
                    for ee in self.events[j:-1]:
                        ee.ind += 1
        # sort based on ind
        self.n_v = len(self.events)
        self.events = sorted(self.events, key=lambda e: (e.ind))

# TrapezoidalDecomposition =============================================================================================


class TrapezoidalDecomposition:
    def __init__(self):
        self.b_poly: Polygon = None
        self.b_obstacle: Obstacle = None
        self.polygons: list[Polygon] = []
        self.obstacles: list[Obstacle] = []
        self.events: list[Event] = []
        self.trapezoids: list[Polygon] = []
        self.n_events: int = 0

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
        self.b_obstacle.inverse_event_types()

    def set_obstacle_polygons(self, polygons: list[Polygon]):
        self.polygons = polygons

    def create_obstacles(self):
        self.obstacles.append(self.b_obstacle)
        for i, p in enumerate(self.polygons):
            self.obstacles.append(Obstacle(i+1, p))

    def create_events(self):
        all_events = [event for obstacle in self.obstacles for event in obstacle.events]
        self.n_events = len(all_events)
        # self.events = all_events
        self.events = sorted(all_events, key=lambda e: (e.xy))
        for i, e in enumerate(self.events):
            e.set_sd_ind(i)

    # decompose_all ============================================================
    def decompose_all(self):
        for i, base_e in enumerate(self.events[1:]):
            #
            ind = i + 1
            if base_e.xy[0] == self.events[0].xy[0]:
                continue

            # Skip border and vertical events
            if base_e.vertical or base_e.done:
                continue

            # find hits of the sweeping line at the base event
            hit_up_y, hit_down_y, hit_up_event, hit_down_event = self.expand_hit(base_e, ind)

            # Update base event with hit results
            base_e.hit_up_y, base_e.hit_up_event = hit_up_y, hit_up_event
            base_e.hit_down_y, base_e.hit_down_event = hit_down_y, hit_down_event

            # update done status of the hit events
            if hit_up_event is not None:
                if hit_up_event.xy[0] == base_e.xy[0]:
                    if hit_up_event.etype == 'down':
                        hit_up_event.done = True
            if hit_down_event is not None:
                if hit_down_event.xy[0] == base_e.xy[0]:
                    if hit_down_event.etype == 'up':
                        hit_down_event.done = True

            # Create trapezoids based on event type
            self.create_trapezoid(base_e, hit_up_event, hit_down_event, hit_up_y, hit_down_y)

    def expand_hit(self, base_e: Event, base_i: int):
        # Initialize hit variables
        hit_up_y, hit_down_y = math.inf, -math.inf
        hit_up_event, hit_down_event = None, None

        # Iterate over events
        for e in self.events[base_i+1:]:
            # Skip events from the same obstacle or fake events
            if (base_e.obst_ind != 0 and base_e.obst_ind == e.obst_ind) or e.fake:
                continue

            # Evaluate event based on base_e type
            success_up, y_up = False, None
            success_down, y_down = False, None

            if base_e.etype in ('up', 'updown'):
                success_up, y_up = self.evel_event_up(base_e, e, hit_up_y)
            if base_e.etype in ('down', 'updown'):
                success_down, y_down = self.eval_event_down(base_e, e, hit_down_y)

            # Update hit variables if successful
            if success_up:
                hit_up_y, hit_up_event = y_up, e
            if success_down:
                hit_down_y, hit_down_event = y_down, e
        #
        return hit_up_y, hit_down_y, hit_up_event, hit_down_event

    def create_trapezoid(self, base_e: Event, hit_up_event: Event, hit_down_event: Event, hit_up_y: float, hit_down_y: float):
        trap1, trap2 = None, None
        if base_e.etype == 'updown' and hit_up_event and hit_down_event:
            if base_e.etype_2 == 'out':
                trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e, base_e.endp_up_e]
                trap2 = [base_e.xy, (base_e.xy[0], hit_down_y), hit_down_event.endp_up_e, base_e.endp_down_e]
            else:
                trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e,
                         hit_down_event.endp_up_e, (base_e.xy[0], hit_down_y)]
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))
        elif base_e.etype == 'up' and hit_up_event:
            trap1 = [base_e.xy, (base_e.xy[0], hit_up_y), hit_up_event.endp_down_e, base_e.endp_up_e]
            hit_up_event.set_endpoint_down_edge((base_e.xy[0], hit_up_y))
        elif base_e.etype == 'down' and hit_down_event:
            trap1 = [base_e.xy, base_e.endp_down_e, hit_down_event.endp_up_e, (base_e.xy[0], hit_down_y)]
            hit_down_event.set_endpoint_up_edge((base_e.xy[0], hit_down_y))
        if trap1:
            self.trapezoids.append(Polygon(trap1))
        if trap2:
            self.trapezoids.append(Polygon(trap2))
        if trap1 is None and trap2 is None:
            print(("Invalid trapezoid creation"))
            # raise ValueError("Invalid trapezoid creation")

    def evel_event_up(self, base_e: Event, e: Event, y_hit: float) -> bool:
        # based even type is "up"
        if e.etype == 'up':
            return False, None

        # reach to higher updown event
        if e.etype == 'updown':
            new_y_hit = e.xy[1]
            if e.xy[0] == base_e.xy[0] and new_y_hit < y_hit:
                return True, new_y_hit
            else:
                return False, None

        # type is "down"
        # edge = e1-e
        e1_ind = e.prev_ind
        e1 = self.obstacles[e.obst_ind].events[e1_ind]

        # check coordinates
        if e1.xy[0] > base_e.xy[0]:
            return False, None
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

        if base_e.sd_ind == 4 and e.sd_ind == 13:
            pass

        # reach to higher updown event
        if e.etype == 'updown':
            new_y_hit = e.xy[1]
            if e.xy[0] == base_e.xy[0] and new_y_hit > y_hit:
                return True, new_y_hit
            else:
                return False, None

        # type is "up"
        # edge = e1-e
        e2_ind = e.next_ind
        e2 = self.obstacles[e.obst_ind].events[e2_ind]

        # check coordinates
        if e2.xy[0] > base_e.xy[0]:
            return False, None
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
        base_e = self.events[self.n_events-2]
        hit_up_event = self.events[self.n_events-1]
        hit_down_event = self.events[self.n_events-3]
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
        plt.subplot(111)

        # plot obstacles
        for obs in self.obstacles:
            x, y = obs.poly.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)

        # display events
        for e in self.events:
            # if e.vertical:
            #     continue

            # texts
            text1 = f"{e.sd_ind}"
            # text1 = f"{e.sd_ind}-[t1:{e.etype}]-[t2:{e.etype_2}]"
            plt.text(e.xy[0], e.xy[1], text1, fontsize=12, color='blue')
            # text2 = f"{e.s_ind}-{e.ind}--[{e.prev_ind},{e.next_ind}]"
            # # text2 = f"{e.ind}--[{e.prev_ind},{e.next_ind}]"
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
        for trap in self.trapezoids:
            x, y = trap.exterior.xy
            plt.fill(x, y, alpha=0.5, fc='yellow', ec='black', linewidth=1)

        #
        plt.show()


# example
def create_bounding():
    # create boarder polygon
    b_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    return b_poly


def create_polygons():
    # create obstacles polygons
    p1 = Polygon([(40, 60), (70, 60), (80, 66), (70, 70), (50, 80), (40, 70)])
    p2 = Polygon([(30, 30), (40, 20), (60, 30), (70, 40), (60, 50), (30, 50)])
    return [p1, p2]


if __name__ == "__main__":
    td = TrapezoidalDecomposition()
    td.set_bounding_polygon(create_bounding())
    td.set_obstacle_polygons(create_polygons())
    td.initiate()
