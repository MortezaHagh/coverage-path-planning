import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

a = defaultdict(set)
a[1].update([2, 3])
a[1].update([1, 3])
print(a)
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
