import dataclasses


class Test1:
    a: int = 1

    def __init__(self):
        self.b = 1


test = Test1()
print(test.a)


@dataclasses.dataclass
class Point:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    @property
    def xy(self):
        return (self.x, self.y)


p1 = Point()
p2 = Point(1, 2)
p3 = Point(x=3, y=4)
p4 = Point(y=5)
p5 = Point(5)

print([p1, p2, p3, p4, p5])
p1.x = 10
print(p1)
print(p3.xy)


# todo to delete
# def plot_connections_arrow(self, ax):
#     start_end_points = get_start_end_points(self.paths_px)
#     ends = [end for _, end in start_end_points]
#     starts = [start for start, _ in start_end_points]
#     ends = np.array(ends, dtype=np.int32)
#     starts = np.array(starts, dtype=np.int32)
#     # fig, ax = plt.subplots(1, 1)
#     for start, end in zip(starts[1:], ends[:-1]):
#         ax.annotate('', xy=(start), xytext=(end), arrowprops=dict(arrowstyle='->,head_width=0.4', color='gray', lw=1.5, ls="--"))
#         # ax.plot([end[0], start[0]], [end[1], start[1]], 'k--', linewidth=0.5)
