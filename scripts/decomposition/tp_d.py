""" Trapezoidal decomposition """
from shapely import Polygon, Point
import matplotlib.pyplot as plt


class Event:
    def __init__(self, ind, v):
        self.etype = "normal"
        self.ind = ind
        self.xy = v

    def update_ind(self, ind):
        self.ind = ind


class Obstacle:
    def __init__(self, poly: Polygon):
        self.poly: Polygon = poly
        self.events: Event = None
        #
        self.create_events()
        self.process_events_1()
        # self.print_events()

    def create_events(self):
        vertices = self.poly.exterior.coords[:-1]
        vertices = sorted(vertices, key=lambda x: (x[0], x[1]))
        self.events = [Event(i, v) for i, v in enumerate(vertices)]

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

    def print_events(self):
        for e in self.events:
            print(f"Event {e.ind}: {e.xy} {e.etype}")


class TrapezoidalDecomposition:
    def __init__(self):
        self.boarder: Polygon = None
        self.polygons: list[Polygon] = []
        self.obstacles: list[Obstacle] = []
        self.events: list[Event] = []

    def initiate(self):
        self.create_polygons()
        self.create_obstacles()
        self.create_events()
        self.visualize()

    def create_polygons(self):

        self.boarder = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        p1 = Polygon([(40, 60), (70, 60), (70, 70), (50, 80), (40, 70)])
        p2 = Polygon([(30, 30), (40, 20), (60, 30), (60, 50), (30, 50)])
        self.polygons = [p1, p2]

    def create_obstacles(self):
        for p in self.polygons:
            self.obstacles.append(Obstacle(p))

    def create_events(self):
        all_events = [event for obstacle in self.obstacles for event in obstacle.events]
        self.events = sorted(all_events, key=lambda e: (e.xy))
        for i, e in enumerate(self.events):
            e.update_ind(i)

    def visualize(self):
        # plot boarder
        plt.subplot(111)
        x, y = self.boarder.exterior.xy
        plt.plot(x, y, color='black', linewidth=2)

        # plot obstacles
        for obs in self.obstacles:
            x, y = obs.poly.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)

        # show vertices index
        for e in self.events:
            text = f"{e.ind}-{e.etype}"
            plt.text(e.xy[0], e.xy[1], text, fontsize=12, color='blue')

        #
        plt.show()


if __name__ == "__main__":
    td = TrapezoidalDecomposition()
    td.initiate()
