""" obstacles for decomposition """
import math
from shapely import Polygon, Point
from scripts.boustrophedon.events import Event


class Obstacle:
    ns: str
    ind: int
    poly: Polygon
    events: list[Event]
    s_events: list[Event]
    n_v: int

    def __init__(self, ind: int, poly: Polygon):
        self.ns = "Obstacle"
        self.ind = ind
        self.poly = poly
        self.events = []
        self.s_events = []
        self.n_v = len(self.poly.exterior.coords)-1
        #
        self.create_events()
        self.eval_events_line()
        self.create_fake_events()
        self.create_sorted_events()
        self.eval_concave()
        if self.ind == 0:
            self.process_events_bounding()
        else:
            self.process_events()
        self.process_fake_events()
        # self.print_events()

    def create_events(self):
        vertices = self.poly.exterior.coords[:-1]
        self.events = [Event(self.ind, i, v) for i, v in enumerate(vertices)]

    def eval_events_line(self):
        rem = []
        for i, e in enumerate(self.events[0:self.n_v-2]):
            if i in rem:
                continue
            i1 = i+1
            i2 = i+2
            dx1 = e.xy[0] - self.events[i1].xy[0]
            dx2 = e.xy[0] - self.events[i2].xy[0]
            if dx1 == 0 and dx2 == 0:
                rem.append(i1)
                while dx2 == 0 and i2 < self.n_v-1:
                    rem.append(i2)
                    i2 += 1
                    dx2 = self.events[i2].xy[0] - e.xy[0]
                rem.pop()

        # update events
        ie = 0
        events = []
        for i, e in enumerate(self.events):
            if i not in rem:
                e.ind = ie
                events.append(e)
                ie += 1
        self.events = events
        self.n_v = len(self.events)
        # update poly
        self.poly = Polygon([Point(e.xy[0], e.xy[1]) for e in self.events])

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
                fake_event.etype = 'updown'
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
            e.set_s_ind(i)
            prev_ind = (self.s_events[i].ind - 1) % (self.n_v)
            next_ind = (self.s_events[i].ind + 1) % (self.n_v)
            e.set_prev_e(self.events[prev_ind])
            e.set_next_e(self.events[next_ind])

    def eval_concave(self) -> bool:
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
                    e.etype_2 = 'in'
                elif self.poly.contains(Point(x-step, y)):
                    e.etype_2 = 'out'
                else:
                    raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [fake]")
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
                        raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [in-out-2]")
                elif c_up:
                    e.etype = 'up'
                elif c_down:
                    e.etype = 'down'
                else:
                    raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [up-down]")
            else:
                raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [in-out]")

            # set endpoints
            if e.etype == 'updown':
                e.set_endpoint_down_edge(prev_e.xy)
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'up':
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'down':
                e.set_endpoint_down_edge(prev_e.xy)
            else:
                raise ValueError(f"[{self.ns}][process_events_bounding] Invalid event type [e.etype]")

            # detect skipped sweep events
            if e.etype_2 == 'in':
                if e.convex:
                    e.etype_3 = "ivex"
                    e.skip_sweep = True
                    #
                    if next_e.xy[0] < prev_e.xy[0]:
                        next_e.has_parent = False
                    else:
                        prev_e.has_parent = False
            elif e.etype_2 == 'out' and e.convex:
                e.etype_3 = "ovex"
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
                    e.etype_2 = 'in'
                elif self.poly.contains(Point(x-step, y)):
                    e.etype_2 = 'out'
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
                e.etype = 'updown'
                if self.poly.contains(mid_p):
                    e.etype_2 = 'in'
                else:
                    e.etype_2 = 'out'
            elif (x > prev_e.xy[0] and x > next_e.xy[0]):
                e.etype = 'updown'
                if self.poly.contains(mid_p):
                    e.etype_2 = 'out'
                else:
                    e.etype_2 = 'in'
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
                        e.etype_2 = 'in'
                    elif self.poly.contains(Point(x-step, y)):
                        e.etype_2 = 'out'
                    else:
                        raise ValueError(f"[{self.ns}][process_events] Invalid event type [in-out-2]")
                elif c_up:
                    e.etype = 'down'
                elif c_down:
                    e.etype = 'up'
                else:
                    raise ValueError(f"[{self.ns}][process_events] Invalid event type [up-down]")
            else:
                raise ValueError(f"[{self.ns}][process_events] Invalid event type [in-out]")

            #
            if e.etype == 'updown' and e.etype_2 is None:
                raise ValueError(f"[{self.ns}][process_events] Invalid event type [updown]")

            # set endpoints
            if e.etype == 'updown':
                e.set_endpoint_down_edge(prev_e.xy)
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'up':
                e.set_endpoint_up_edge(next_e.xy)
            elif e.etype == 'down':
                e.set_endpoint_down_edge(prev_e.xy)
            else:
                raise ValueError(f"[{self.ns}][process_events] Invalid event type [e.etype]")

            # detect skipped sweep events
            if e.etype_2 == 'out':
                if e.concave:
                    e.etype_3 = "ocave"
                    e.skip_sweep = True
                    #
                    if next_e.xy[0] < prev_e.xy[0]:
                        next_e.has_parent = False
                    else:
                        prev_e.has_parent = False
            elif e.etype_2 == 'in' and e.concave:
                e.etype_3 = "icave"
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
            if e.etype_2 == 'in':
                case_in = True
            elif e.etype_2 == 'out':
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
        if up_e.etype == 'down' and down_e.etype == 'updown':
            up_e.skip_sweep = True
            down_e.skip_sweep = False
            down_e.etype = 'down'
            up_e.skip_hit = True
        elif down_e.etype == 'up' and up_e.etype == 'updown':
            down_e.skip_sweep = True
            up_e.skip_sweep = False
            up_e.etype = 'up'
            down_e.skip_hit = True
        elif up_e.etype == 'updown' and down_e.etype == 'updown':
            down_e.skip_sweep = False
            down_e.etype = 'down'
            up_e.skip_sweep = False
            up_e.etype = 'up'
        elif up_e.etype == 'down' and down_e.etype == 'up':
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
        if up_e.etype == 'down' and down_e.etype == 'updown':
            up_e.skip_sweep = False
            down_e.skip_sweep = True
            down_e.skip_hit = True
        elif down_e.etype == 'up' and up_e.etype == 'updown':
            down_e.skip_sweep = False
            up_e.skip_sweep = True
            up_e.skip_hit = True
        elif up_e.etype == 'updown' and down_e.etype == 'updown':
            e.skip_sweep = False
            e.etype_2 = 'in'
            down_e.skip_sweep = True
            up_e.skip_sweep = True
            down_e.skip_hit = True
            up_e.skip_hit = True
        elif up_e.etype == 'down' and down_e.etype == 'up':
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
        if up_e.etype == 'updown' and down_e.etype == 'down':
            down_e.skip_sweep = True
            up_e.skip_sweep = False
            up_e.etype = 'down'
        elif up_e.etype == 'up' and down_e.etype == 'updown':
            down_e.skip_sweep = False
            up_e.skip_sweep = True
            down_e.etype = 'up'
        elif up_e.etype == 'updown' and down_e.etype == 'updown':
            up_e.skip_sweep = True
            down_e.skip_sweep = False
            down_e.etype = 'up'
            up_e.etype = 'down'
        elif up_e.etype == 'up' and down_e.etype == 'down':
            up_e.skip_sweep = True
            down_e.skip_sweep = True
            e.skip_sweep = False
        else:
            raise ValueError(f"[{self.ns}][process_fake_in_events] Invalid event types")

    def process_fake_out_events(self, e: Event):
        up_e = e.prev_e if e.prev_e.xy[1] > e.xy[1] else e.next_e
        down_e = e.next_e if e.prev_e.xy[1] > e.xy[1] else e.prev_e
        #
        if up_e.etype == 'up' and down_e.etype == 'updown':
            up_e.skip_sweep = False
            down_e.skip_sweep = True
            down_e.etype = 'up'
        elif down_e.etype == 'down' and up_e.etype == 'updown':
            down_e.skip_sweep = False
            up_e.skip_sweep = True
            up_e.etype = 'down'
        elif up_e.etype == 'up' and down_e.etype == 'down':
            up_e.skip_sweep = False
            down_e.skip_sweep = False
        elif up_e.etype == 'updown' and down_e.etype == 'updown':
            up_e.skip_sweep = True
            down_e.skip_sweep = True
            up_e.etype = 'up'
            down_e.etype = 'down'
        else:
            raise ValueError(f"[{self.ns}][process_fake_in_events] Invalid event types")

    def inverse_event_types(self):
        for e in self.events:
            e.prev_e, e.next_e = e.next_e, e.prev_e
            # swpap endpoints
            e.end_up_ed_xy, e.end_down_ed_xy = e.end_down_ed_xy, e.end_up_ed_xy

    def print_events(self):
        for e in self.events:
            print(f"[self.ns]: Event {e.ind}: {e.xy} {e.etype}")
