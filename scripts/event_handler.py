import threading
from pynput.keyboard import Listener, Key
import cv2 as cv
import numpy as np
from map_data import MapData, Configs, mod_angle


class Rectangle:
    def __init__(self, data):
        #
        self.ns = "Rectangle"
        x = data[0]
        y = data[1]
        rotation = data[4]
        self.width = data[2]
        self.height = data[3]

        #
        self.x = np.abs(x).astype(int)
        self.y = np.abs(y).astype(int)
        self.rotation = mod_angle(rotation)


class VisualRectangle:
    def __init__(self, data: list):
        self.ns = "VisualRectangle"

        # configs
        self.line_width = 2
        self.point_radius = 10

        self.rectangle = Rectangle(data=data)


class RecEventHandler:
    vis_obj: list[VisualRectangle]

    def __init__(self, map_data: MapData, configs: Configs):
        #
        self.ns = "RecEventHandler"
        self.wind_name = "Select"

        #
        self.md = map_data
        self.cf = configs

        #
        self.ctrl_down = False

        #
        self.vis_obj = None  # []

    def mouse_handle(self, event, x, y, flags, param):
        mx, my = self.md.point_to_map(x, y)

        #
        if event == cv.EVENT_LBUTTONDBLCLK:
            print(f"[{self.ns}]: [x,y], [{(x, y)}]")
            print(f"[{self.ns}]: [mx,my], [{(mx, my)}]")
            self.on_lb_db_click(mx, my, flags)

        if event == cv.EVENT_LBUTTONUP:
            self.on_button_up(x, y, flags)

    def on_lb_db_click(self, x, y, flags):
        print(f"[{self.ns}]: lb_db clicked")
        params = [x, y, 50, 50, 0]
        self.vis_obj = VisualRectangle(data=params)

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
        self.rec_ev_handler = RecEventHandler(map_data, configs)

        #  set mouse callback
        cv.setMouseCallback(self.wind_name, self.rec_ev_handler.mouse_handle)
        listener_thread = threading.Thread(
            target=self.key_listener, args=(self.rec_ev_handler,))
        listener_thread.start()

        # Main Loop
        cv.imshow(self.wind_name, self.md.up_map)
        while True:
            cv.imshow(self.wind_name, self.md.up_map)

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
