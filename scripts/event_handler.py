import copy
import threading
from pynput.keyboard import Listener, Key
import cv2 as cv
import numpy as np
from shapely.geometry import box, Point

from my_utils import mod_angle
from map_data import MapData, Configs


class Rectangle:
    bbox: box

    def __init__(self, data):
        #
        self.ns = "Rectangle"
        xc = data[0]
        yc = data[1]
        rot = data[4]
        w = data[2]
        h = data[3]

        #
        self.width = w
        self.height = h
        self.xc = xc
        self.yc = yc
        self.center = (xc, yc)
        #
        self.xc_i = np.abs(xc).astype(int)
        self.yc_i = np.abs(yc).astype(int)
        self.rot = mod_angle(rot)

        # create box
        bbox = box(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        self.box = bbox

        # Get the coordinates of the vertices
        vertices = list(bbox.exterior.coords)
        vertices = np.array(vertices, dtype=np.int32)
        vertices2 = vertices.reshape((-1, 1, 2))

        #
        self.vertices = vertices
        self.vertices2 = vertices2


class VisualRectangle:
    rect: Rectangle

    def __init__(self, data: list):
        self.ns = "VisualRectangle"

        # configs
        self.line_width = 10
        self.point_radius = 40
        self.default_color = (0, 0, 255)
        self.highlight_color = (255, 0, 0)
        self.highlight_color_fill = (255, 255, 255)

        #
        self.highlight = False

        #
        self.rect = Rectangle(data=data)

    def is_in_box(self, x, y):
        point = Point(x, y)
        is_inside = self.rect.box.contains(point)
        if is_inside:
            self.highlight = True
        else:
            self.highlight = False

    def draw(self, img: np.ndarray):
        verts = self.rect.vertices2
        color = self.default_color
        if self.highlight:
            # update color
            color = self.highlight_color
            # draw points
            cv.circle(img, self.rect.center, self.point_radius, self.highlight_color, -1)
            for vertex in self.rect.vertices:
                cv.circle(img, tuple(vertex), self.point_radius, self.highlight_color, -1)
        # draw box
        cv.polylines(img, [verts], True, color, self.line_width,)


#
# ============================== RecEventHandler ===============================
#


class RecEventHandler:
    vis_obj: VisualRectangle
    img: np.ndarray

    def __init__(self, map_img: np.ndarray, configs: Configs):
        #
        self.ns = "RecEventHandler"
        self.wind_name = "Select"

        #
        self.cf = configs
        self.map_img = map_img
        self.canvas = np.zeros_like(map_img, dtype=np.uint8)
        # self.md = map_data
        # self.img = map_data.up_rgb_map

        #
        self.ctrl_down = False

        #
        self.vis_obj = None  # []

    def mouse_handle(self, event, x, y, flags, param):
        # mx, my = self.md.point_to_map(x, y)

        #
        if event == cv.EVENT_MOUSEMOVE:
            self.on_mouse_move(x, y, flags)

        if event == cv.EVENT_LBUTTONDBLCLK:
            self.on_lb_db_click(x, y, flags)

        if event == cv.EVENT_LBUTTONDOWN:
            self.on_button_down(x, y, flags)

        if event == cv.EVENT_LBUTTONUP:
            self.on_button_up(x, y, flags)

    def on_mouse_move(self, x, y, flags):
        if self.vis_obj is None:
            return
        self.vis_obj.is_in_box(x, y)

    def on_lb_db_click(self, x, y, flags):
        print(f"[{self.ns}]: lb_db clicked")
        print(f"[{self.ns}]: [x,y], [{(x, y)}]")
        params = [x, y, 200, 200, 0]
        self.vis_obj = VisualRectangle(data=params)

    def on_button_down(self, x, y, flags):
        if self.vis_obj is None:
            return
        if self.vis_obj.highlight:
            print("ooooooooooo")

    def on_button_up(self, x, y, flags):
        pass

    def on_press(self, key):
        print(f"[{self.ns}]: a key is down")

        if key == Key.ctrl:
            self.ctrl_down = True
        elif key == Key.esc or key == Key.enter:
            return False

        if key == Key.ctrl:
            print(f"[{self.ns}]: crtls key is down")

    def on_release(self, key):
        print(f"[{self.ns}]: key is released")
        if key == Key.esc or key == Key.enter:
            return False

        if key == Key.ctrl:
            self.ctrl_down = False

    def update_image(self):
        self.canvas[:, :] = self.map_img
        if self.vis_obj is None:
            return
        self.vis_obj.draw(self.canvas)

#
# ============================== MainEventHandler ==============================
#


class MainEventHandler:
    def __init__(self, map_data: MapData, configs: Configs):
        #
        self.ns = "MainEventHandler"
        self.wind_name = "Select"

        #
        self.md = map_data
        self.cf = configs

        # init window
        self.init_wind()

        # init rectangle event handler
        rec_ev_handler = RecEventHandler(map_data.up_rgb_map, configs)

        #  set mouse callback
        cv.setMouseCallback(self.wind_name, rec_ev_handler.mouse_handle)
        listener_thread = threading.Thread(target=self.key_listener, args=(rec_ev_handler,))
        listener_thread.start()

        # Main Loop
        cv.imshow(self.wind_name, self.md.up_rgb_map)
        while True:
            rec_ev_handler.update_image()
            cv.imshow(self.wind_name, rec_ev_handler.canvas)

            c2 = cv.getWindowProperty(self.wind_name, cv.WND_PROP_VISIBLE) < 1
            if c2 or (cv.waitKey(20) & 0xFF == 13 or cv.waitKey(20) & 0xFF == 27):
                break
        #
        listener_thread.join()
        cv.destroyAllWindows()

    def init_wind(self):
        cv.namedWindow(self.wind_name, cv.WINDOW_GUI_NORMAL)

    def key_listener(self, handler: RecEventHandler):
        with Listener(on_press=handler.on_press, on_release=handler.on_release) as listener:
            listener.join()
