import threading
from pynput.keyboard import Listener
import cv2 as cv
from scripts.map_data import MapData, Configs
from scripts.gui.rect_event_hl import RecEventHandler
from scripts.gui.visuals import Rectangle, VisualRectangle


class District:
    rec_ev_hl: RecEventHandler

    def __init__(self, map_data: MapData, configs: Configs):
        #
        self.ns = "District"
        self.wind_name = "Select"

        #
        self.md = map_data
        self.cf = configs

        # init window
        self.init_wind()

        #
        if self.cf.whole_map:
            print(f"[{self.ns}]: whole map is selected as district for CPP!")
            return None
        else:
            print(f"[{self.ns}]: Selecting district for CPP ... ")

        # init rectangle event handler
        rec_ev_hl = RecEventHandler(map_data.up_rgb_map, configs)
        self.rec_ev_hl = rec_ev_hl

        #  set mouse callback
        cv.setMouseCallback(self.wind_name, rec_ev_hl.mouse_handle)
        listener_thread = threading.Thread(target=self.key_listener, args=(rec_ev_hl,))
        listener_thread.start()

        # Main Loop
        cv.imshow(self.wind_name, self.md.up_rgb_map)
        while True:
            rec_ev_hl.update_image()
            cv.imshow(self.wind_name, rec_ev_hl.canvas)

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

    def get_district_vis(self) -> VisualRectangle:
        if self.cf.whole_map:
            print(f"[{self.ns}]: whole map is selected as district for CPP! There is no rec_ev_hl!")
            return None
        return self.rec_ev_hl.get_main_obj()

    def get_district_rect(self) -> Rectangle:
        if not self.cf.whole_map:
            return self.rec_ev_hl.get_main_obj().get_rect()

        # create district based on map data
        xc = self.md.up_c_w
        yc = self.md.up_c_h
        rot = 0
        w = self.md.updated_w
        h = self.md.updated_h
        data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        return Rectangle(data)
