import threading
from pynput.keyboard import Listener
import cv2 as cv
from scripts.map_data import MapData, Configs
from scripts.gui.rect_event_hl import RecEventHandler


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
