""" obstacles class for decomposition """
import math
from shapely import Polygon, Point
from covpp.coverage_planning.boustrophedon.events import Event


class Obstacle:
    """Encapsulate a polygonal obstacle and its decomposition events.

    Each obstacle generates a sequence of sweep-line events that drive the
    trapezoidal decomposition. The class pre-processes vertices, injects fake
    events for vertical edges, and evaluates event types (up/down/in/out) used
    by :class:`CellDecomposition`.
    """

    def __init__(self, ind: int, poly: Polygon):
        self.ns: str = "Obstacle"
        #
        self.ind: int = ind
        self.poly: Polygon = poly   # shapely polygon
        self.events: list[Event] = []
        self.s_events: list[Event] = []
        self.n_v: int = 0
        #
        vertices = self.poly.exterior.coords[:-1]
        filtered_vertices = self.eval_vertices_line(vertices)
        self.create_events(filtered_vertices)
        self.create_fake_events()
        self.create_sorted_events()
        self.eval_concave()
        if self.ind == 0:
            self.process_events_bounding()
        else:
            self.process_events()
        self.process_fake_events()
        # self.print_events()

    def create_events(self, filtered_vertices: list[tuple[float, float]]):
        self.events = [Event(self.ind, i, v) for i, v in enumerate(filtered_vertices)]
        self.n_v = len(self.events)
        self.poly = Polygon([Point(e.xy[0], e.xy[1]) for e in self.events])

    def eval_vertices_line(self, vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Remove vertices that are on the same vertical line between two other vertices, then create events. """
        n_v = len(vertices)
        rem = set()
        i = 0
        while i < n_v - 2:
            x0 = vertices[i][0]
            x1 = vertices[i + 1][0]
            x2 = vertices[i + 2][0]
            if x0 == x1 == x2:
                rem.add(i + 1)
                j = i + 2
                while j < n_v - 1 and vertices[j][0] == x0:
                    rem.add(j)
                    j += 1
                # Remove last added (to match original logic)
                rem.discard(j - 1)
                i = j - 1
            else:
                i += 1

        # Filter vertices
        filtered_vertices = [v for idx, v in enumerate(vertices) if idx not in rem]
        return filtered_vertices

    def create_fake_events(self):
        new_events = []
        for i, e in enumerate(self.events[:self.n_v]):
            new_events.append(e)
            j = i+1
            if j == self.n_v:
                j = 0
            e2 = self.events[j]
            if e.xy[0] == e2.xy[0]:
                e.vertical = True
                e2.vertical = True
                xy = (e.xy[0], (e.xy[1] + e2.xy[1]) / 2)
                # create fake event
                fake_event = Event(self.ind, j, xy)
                fake_event.etype1_updown = 'updown'
                fake_event.fake = True
                if j == 0:
                    new_events.insert(0, fake_event)
                else:
                    new_events.append(fake_event)
        # add fake events to the events list
        for i, e in enumerate(new_events):
            e.ind = i

        # sort based on ind
        self.events = new_events  # sorted(new_events, key=lambda e: (e.ind))
        self.n_v = len(self.events)

    def create_sorted_events(self):
        self.s_events = sorted(self.events, key=lambda e: (e.xy[0], e.xy[1]))
        for i, e in enumerate(self.s_events):
            e.set_sorted_ind(i)
            prev_ind = (self.s_events[i].ind - 1) % (self.n_v)
            next_ind = (self.s_events[i].ind + 1) % (self.n_v)
            e.set_prev_e(self.events[prev_ind])
            e.set_next_e(self.events[next_ind])

    def eval_concave(self) -> None:
        for e in self.events:
            x, y = e.xy
            prev_x, prev_y = e.prev_e.xy
            next_x, next_y = e.next_e.xy
            # Check if the event is concave
            e.concave = (prev_x - x) * (next_y - y) - (prev_y - y) * (next_x - x) > 0
            e.convex = not e.concave

    def process_events_bounding(self):
        for e in self.events:
            # swap prev and next indices
            e.prev_e, e.next_e = e.next_e, e.prev_e

            # x, y, preve, nexte
            x, y = e.xy
            next_e = e.next_e
            prev_e = e.prev_e

            # set in-out for fake events
            step = 0.2
            if e.fake:
                if self.poly.contains(Point(x+step, y)):
                    e.etype2_inout = 'in'
                elif self.poly.contains(Point(x-step, y)):
                    e.etype2_inout = 'out'
                else:
                    raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [fake]")
                continue

            # detect event types: up, down, updown, in, out
            if x < prev_e.xy[0] and x < next_e.xy[0]:
                e.etype1_updown = 'updown'
                e.etype2_inout = 'in'
            elif x > prev_e.xy[0] and x > next_e.xy[0]:
                e.etype1_updown = 'updown'
                e.etype2_inout = 'out'
            elif prev_e.xy[0] < x < next_e.xy[0]:
                e.etype1_updown = 'down'
            elif next_e.xy[0] < x < prev_e.xy[0]:
                e.etype1_updown = 'up'
            elif x == prev_e.xy[0] or x == next_e.xy[0]:
                c_up = self.poly.contains(Point(x, y + step)) or self.poly.touches(Point(x, y + step))
                c_down = self.poly.contains(Point(x, y - step)) or self.poly.touches(Point(x, y - step))
                if c_up and c_down:
                    e.etype1_updown = 'updown'
                    if self.poly.contains(Point(x+step, y)):
                        e.etype2_inout = 'out'
                    elif self.poly.contains(Point(x-step, y)):
                        e.etype2_inout = 'in'
                    else:
                        raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [in-out-2]")
                elif c_up:
                    e.etype1_updown = 'up'
                elif c_down:
                    e.etype1_updown = 'down'
                else:
                    raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [up-down]")
            else:
                raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [in-out]")

            # set endpoints
            if e.etype1_updown == 'updown':
                e.set_endpoint_down_edge(prev_e.xy)
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype1_updown == 'up':
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype1_updown == 'down':
                e.set_endpoint_down_edge(prev_e.xy)
            else:
                raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [e.etype1_updown]")

            # detect skipped sweep events
            if e.etype2_inout == 'in':
                if e.convex:
                    e.etype3_io_conc = "ivex"
                    e.skip_sweep = True
                    #
                    if next_e.xy[0] < prev_e.xy[0]:
                        next_e.has_parent = False
                    else:
                        prev_e.has_parent = False
            elif e.etype2_inout == 'out' and e.convex:
                e.etype3_io_conc = "ovex"
                if not e.vertical:
                    e.direct_trap = True
                    e.has_adj = False
                e.x_max = max(prev_e.xy[0], next_e.xy[0])

    def process_events(self):
        for e in self.events:
            # x, y, preve, nexte
            x, y = e.xy
            next_e = e.next_e
            prev_e = e.prev_e

            # set in-out for fake events
            step = 1.0
            if e.fake:
                if self.poly.contains(Point(x+step, y)):
                    e.etype2_inout = 'in'
                elif self.poly.contains(Point(x-step, y)):
                    e.etype2_inout = 'out'
                else:
                    raise ValueError(f"[{self.ns}][process_events] Invalid event type [fake]")
                continue

            # detect event types: up, down, updown, in, out
            mid_x = (prev_e.xy[0] + next_e.xy[0]) / 2
            mid_y = (prev_e.xy[1] + next_e.xy[1]) / 2
            theta = math.atan2(mid_y - y, mid_x - x)
            mid_x = x + step * math.cos(theta)
            mid_y = y + step * math.sin(theta)
            mid_p = Point(mid_x, mid_y)
            if (x < prev_e.xy[0] and x < next_e.xy[0]):
                e.etype1_updown = 'updown'
                if self.poly.contains(mid_p):
                    e.etype2_inout = 'in'
                else:
                    e.etype2_inout = 'out'
            elif (x > prev_e.xy[0] and x > next_e.xy[0]):
                e.etype1_updown = 'updown'
                if self.poly.contains(mid_p):
                    e.etype2_inout = 'out'
                else:
                    e.etype2_inout = 'in'
            elif prev_e.xy[0] < x < next_e.xy[0]:
                e.etype1_updown = 'down'
            elif next_e.xy[0] < x < prev_e.xy[0]:
                e.etype1_updown = 'up'
            elif x == prev_e.xy[0] or x == next_e.xy[0]:
                c_up = self.poly.contains(Point(x, y + step)) or self.poly.touches(Point(x, y + step))
                c_down = self.poly.contains(Point(x, y - step)) or self.poly.touches(Point(x, y - step))
                if c_up and c_down:
                    e.etype1_updown = 'updown'
                    if self.poly.contains(Point(x+step, y)):
                        e.etype2_inout = 'in'
                    elif self.poly.contains(Point(x-step, y)):
                        e.etype2_inout = 'out'
                    else:
                        raise ValueError(f"[{self.ns}][process_events] Invalid event type [in-out-2]")
                elif c_up:
                    e.etype1_updown = 'down'
                elif c_down:
                    e.etype1_updown = 'up'
                else:
                    raise ValueError(f"[{self.ns}][process_events] Invalid event type [up-down]")
            else:
                raise ValueError(f"[{self.ns}][process_events] Invalid event type [in-out]")

            #
            if e.etype1_updown == 'updown' and e.etype2_inout is None:
                raise ValueError(f"[{self.ns}][process_events] Invalid event type [updown]")

            # set endpoints
            if e.etype1_updown == 'updown':
                e.set_endpoint_down_edge(prev_e.xy)
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype1_updown == 'up':
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype1_updown == 'down':
                e.set_endpoint_down_edge(prev_e.xy)
            else:
                raise ValueError(f"[{self.ns}][process_events] Invalid event type [e.etype1_updown]")

            # detect skipped sweep events
            if e.etype2_inout == 'out':
                if e.concave:
                    e.etype3_io_conc = "ocave"
                    e.skip_sweep = True
                    #
                    if next_e.xy[0] < prev_e.xy[0]:
                        next_e.has_parent = False
                    else:
                        prev_e.has_parent = False
            elif e.etype2_inout == 'in' and e.concave:
                e.etype3_io_conc = "icave"
                if not e.vertical:
                    e.direct_trap = True
                    e.has_adj = False
                e.x_max = max(prev_e.xy[0], next_e.xy[0])

    def process_fake_events(self):
        for e in self.events:
            if not e.fake:
                continue
            #
            e.skip_hit = True
            e.skip_sweep = True

            #
            case_in = False
            if e.etype2_inout == 'in':
                case_in = True
            elif e.etype2_inout == 'out':
                case_in = False
            else:
                raise ValueError(f"[{self.ns}][process_fake_events] Invalid event type [fake]")

            #
            if case_in:
                if e.obst_ind == 0:
                    self.process_fake_in_events_b(e)
                else:
                    self.process_fake_in_events(e)
            else:
                if e.obst_ind == 0:
                    self.process_fake_out_events_b(e)
                else:
                    self.process_fake_out_events(e)

    def process_fake_in_events_b(self, e: Event):
        up_e = e.prev_e if e.prev_e.xy[1] > e.xy[1] else e.next_e
        down_e = e.next_e if e.prev_e.xy[1] > e.xy[1] else e.prev_e
        #
        if up_e.etype1_updown == 'down' and down_e.etype1_updown == 'updown':
            up_e.skip_sweep = True
            down_e.skip_sweep = False
            down_e.etype1_updown = 'down'
            up_e.skip_hit = True
        elif down_e.etype1_updown == 'up' and up_e.etype1_updown == 'updown':
            down_e.skip_sweep = True
            up_e.skip_sweep = False
            up_e.etype1_updown = 'up'
            down_e.skip_hit = True
        elif up_e.etype1_updown == 'updown' and down_e.etype1_updown == 'updown':
            down_e.skip_sweep = False
            down_e.etype1_updown = 'down'
            up_e.skip_sweep = False
            up_e.etype1_updown = 'up'
        elif up_e.etype1_updown == 'down' and down_e.etype1_updown == 'up':
            down_e.skip_sweep = True
            up_e.skip_sweep = True
            down_e.skip_hit = True
            up_e.skip_hit = True
            #
            e.has_adj = False
            #
            n_e = up_e.next_e
            p_e = down_e.prev_e
            if n_e.xy[0] < p_e.xy[0]:
                n_e.has_parent = False
            else:
                p_e.has_parent = False
        else:
            raise ValueError(f"[{self.ns}][process_fake_in_events] Invalid event types")

    def process_fake_out_events_b(self, e: Event):
        up_e = e.prev_e if e.prev_e.xy[1] > e.xy[1] else e.next_e
        down_e = e.next_e if e.prev_e.xy[1] > e.xy[1] else e.prev_e
        #
        if up_e.etype1_updown == 'down' and down_e.etype1_updown == 'updown':
            up_e.skip_sweep = False
            down_e.skip_sweep = True
            down_e.skip_hit = True
        elif down_e.etype1_updown == 'up' and up_e.etype1_updown == 'updown':
            down_e.skip_sweep = False
            up_e.skip_sweep = True
            up_e.skip_hit = True
        elif up_e.etype1_updown == 'updown' and down_e.etype1_updown == 'updown':
            e.skip_sweep = False
            e.etype2_inout = 'in'
            down_e.skip_sweep = True
            up_e.skip_sweep = True
            down_e.skip_hit = True
            up_e.skip_hit = True
        elif up_e.etype1_updown == 'down' and down_e.etype1_updown == 'up':
            up_e.skip_sweep = True
            down_e.skip_sweep = False
            #
            down_e.has_adj = False
        else:
            raise ValueError(f"[{self.ns}][process_fake_in_events] Invalid event types")

    def process_fake_in_events(self, e: Event):
        up_e = e.prev_e if e.prev_e.xy[1] > e.xy[1] else e.next_e
        down_e = e.next_e if e.prev_e.xy[1] > e.xy[1] else e.prev_e
        #
        if up_e.etype1_updown == 'updown' and down_e.etype1_updown == 'down':
            down_e.skip_sweep = True
            up_e.skip_sweep = False
            up_e.etype1_updown = 'down'
        elif up_e.etype1_updown == 'up' and down_e.etype1_updown == 'updown':
            down_e.skip_sweep = False
            up_e.skip_sweep = True
            down_e.etype1_updown = 'up'
        elif up_e.etype1_updown == 'updown' and down_e.etype1_updown == 'updown':
            up_e.skip_sweep = True
            down_e.skip_sweep = False
            down_e.etype1_updown = 'up'
            up_e.etype1_updown = 'down'
        elif up_e.etype1_updown == 'up' and down_e.etype1_updown == 'down':
            up_e.skip_sweep = True
            down_e.skip_sweep = True
            e.skip_sweep = False
        else:
            raise ValueError(f"[{self.ns}][process_fake_in_events] Invalid event types")

    def process_fake_out_events(self, e: Event):
        up_e = e.prev_e if e.prev_e.xy[1] > e.xy[1] else e.next_e
        down_e = e.next_e if e.prev_e.xy[1] > e.xy[1] else e.prev_e
        #
        if up_e.etype1_updown == 'up' and down_e.etype1_updown == 'updown':
            up_e.skip_sweep = False
            down_e.skip_sweep = True
            down_e.etype1_updown = 'up'
        elif down_e.etype1_updown == 'down' and up_e.etype1_updown == 'updown':
            down_e.skip_sweep = False
            up_e.skip_sweep = True
            up_e.etype1_updown = 'down'
        elif up_e.etype1_updown == 'up' and down_e.etype1_updown == 'down':
            up_e.skip_sweep = False
            down_e.skip_sweep = False
        elif up_e.etype1_updown == 'updown' and down_e.etype1_updown == 'updown':
            up_e.skip_sweep = True
            down_e.skip_sweep = True
            up_e.etype1_updown = 'up'
            down_e.etype1_updown = 'down'
        else:
            # todo: analyze this case
            up_e.skip_sweep = True
            down_e.skip_sweep = True
            print(f"[{self.ns}][process_fake_out_events] Error: Unexpected event types.")
            # raise ValueError(f"[{self.ns}][process_fake_in_events] Invalid event types")

    # def inverse_event_types(self):
    #     for e in self.events:
    #         e.prev_e, e.next_e = e.next_e, e.prev_e
    #         # swap endpoints
    #         e.end_up_ed_xy, e.end_down_ed_xy = e.end_down_ed_xy, e.end_up_ed_xy

    def print_events(self):
        for e in self.events:
            print(f"[self.ns]: Event {e.ind}: {e.xy} {e.etype1_updown}")
