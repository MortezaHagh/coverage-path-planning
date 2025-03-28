""" Trapezoidal decomposition """
from shapely import Polygon, Point
import matplotlib.pyplot as plt


class Event:
    def __init__(self, obst_ind, ind, v):
        self.etype: str = "normal"
        self.obst_ind: int = obst_ind
        self.ind: int = ind         # index in the obstacle (ccw)
        self.s_ind: int = None      # sorted index in the obstacles events
        self.sd_ind: int = None     # sorted index in the decomposition
        self.prev_ind: int = None
        self.next_ind: int = None
        self.xy = v

    def set_s_ind(self, ind: int):
        self.s_ind = ind

    def set_sd_ind(self, ind: int):
        self.sd_ind = ind

    def set_prev_ind(self, ind: int):
        self.prev_ind = ind

    def set_next_ind(self, ind: int):
        self.next_ind = ind

# Obstacle ====================================================================


class Obstacle:
    def __init__(self, ind: int, poly: Polygon):
        self.ind: int = ind
        self.poly: Polygon = poly
        self.events: list[Event] = []
        self.s_events: list[Event] = []
        self.n_v: int = len(self.poly.exterior.coords)
        #
        self.create_events()
        self.create_sorted_events()
        self.process_events_1()
        # self.print_events()

    def create_events(self):
        vertices = self.poly.exterior.coords[:-1]
        self.events = [Event(self.ind, i, v) for i, v in enumerate(vertices)]

    def create_sorted_events(self):
        self.s_events = sorted(self.events, key=lambda e: (e.xy[0], e.xy[1]))
        for i, e in enumerate(self.s_events):
            e.set_s_ind(i)
            prev_ind = (self.s_events[i].ind - 1) % self.n_v
            next_ind = (self.s_events[i].ind + 1) % self.n_v
            e.set_prev_ind(prev_ind)
            e.set_next_ind(next_ind)

    def process_events_1(self):
        for e in self.events:
            x, y = e.xy
            step = 1.0  # Set step size as a parameter
            # Check if the point in +y direction is inside or on the boundary of the obstacle
            if self.poly.contains(Point(x, y + step)) or self.poly.touches(Point(x, y + step)):
                e.etype = 'down'
            # Check if the point in -y direction is inside or on the boundary of the obstacle
            elif self.poly.contains(Point(x, y - step)) or self.poly.touches(Point(x, y - step)):
                e.etype = 'up'
            # If neither point is inside or on the boundary of the obstacle
            else:
                e.etype = 'updown'

    def process_events_2(self):
        for i, e1 in enumerate(self.events):
            for e2 in self.events[i+1:]:
                if e1.xy[0] == e2.xy[0]:
                    e1.etype, e2.etype = ('up', 'down') if (e1.xy[1] > e2.xy[1]) else ('down', 'up')

    def inverse_event_types(self):
        for e in self.events:
            if e.etype == 'up':
                e.etype = 'down'
            elif e.etype == 'down':
                e.etype = 'up'

    def print_events(self):
        for e in self.events:
            print(f"Event {e.ind}: {e.xy} {e.etype}")


# TrapezoidalDecomposition =============================================================================================


class TrapezoidalDecomposition:
    def __init__(self):
        self.b_poly: Polygon = None
        self.b_obstacle: Obstacle = None
        self.polygons: list[Polygon] = []
        self.obstacles: list[Obstacle] = []
        self.events: list[Event] = []

    def initiate(self):
        self.create_polygons()
        self.create_obstacles()
        self.create_events()
        self.visualize()

    def create_polygons(self):
        # create boarder polygon
        self.b_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        self.b_obstacle = Obstacle(0, self.b_poly)
        self.b_obstacle.inverse_event_types()
        # create obstacles polygons
        p1 = Polygon([(40, 60), (70, 60), (70, 70), (50, 80), (40, 70)])
        p2 = Polygon([(30, 30), (40, 20), (60, 30), (60, 50), (30, 50)])
        self.polygons = [p1, p2]

    def create_obstacles(self):
        self.obstacles.append(self.b_obstacle)
        for i, p in enumerate(self.polygons):
            self.obstacles.append(Obstacle(i+1, p))

    def create_events(self):
        all_events = [event for obstacle in self.obstacles for event in obstacle.events]
        # self.events = all_events
        self.events = sorted(all_events, key=lambda e: (e.xy))
        for i, e in enumerate(self.events):
            e.set_sd_ind(i)

    def visualize(self):
        # plot boarder
        plt.subplot(111)

        # # plot boarder polygon
        # x, y = self.b_poly.exterior.xy
        # plt.plot(x, y, color='black', linewidth=2)

        # plot obstacles
        for obs in self.obstacles:
            x, y = obs.poly.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)

        # # display boarder_events
        # for e in self.b_obstacle.events:
        #     text1 = f"{e.etype}"
        #     text2 = f"{e.s_ind}-{e.ind}--[{e.prev_ind},{e.next_ind}]"
        #     plt.text(e.xy[0], e.xy[1], text1, fontsize=12, color='blue')
        #     plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

        # display events
        for e in self.events:
            text1 = f"{e.sd_ind}-{e.etype}"
            text2 = f"{e.s_ind}-{e.ind}--[{e.prev_ind},{e.next_ind}]"
            plt.text(e.xy[0], e.xy[1], text1, fontsize=12, color='blue')
            plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

        #
        plt.show()


if __name__ == "__main__":
    td = TrapezoidalDecomposition()
    td.initiate()
