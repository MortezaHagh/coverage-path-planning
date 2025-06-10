import threading
# from pynput.keyboard import Listener
import cv2 as cv
from scripts.base.map_data import MapData
from scripts.base.configs import DistrictConfgs
from scripts.gui.rect_event_hl import RecEventHandler
from scripts.gui.visuals import Rectangle  # , VisualRectangle


class DistrictGen:
    __ns: str
    __wind_name: str
    __md: MapData
    __cfg: DistrictConfgs
    #
    __rec_ev_hl: RecEventHandler

    def __init__(self, map_data: MapData):
        #
        self.__ns = "DistrictEvent"
        self.__wind_name = "Select District"
        print(f" >>>>> [{self.__ns}] Step: Creating district for CPP")

        #
        self.__md = map_data
        self.__cfg = DistrictConfgs()

        #
        if self.__cfg.whole_map:
            print(f"[{self.__ns}]: whole map is selected as district for CPP!")
            return None
        print(f"[{self.__ns}]: Selecting district for CPP ... ")

        # init window
        self.init_wind()

        # init rectangle event handler
        rec_ev_hl = RecEventHandler(map_data.up_rgb_map)
        self.__rec_ev_hl = rec_ev_hl

        #  set mouse callback
        cv.setMouseCallback(self.__wind_name, rec_ev_hl.mouse_handle)
        listener_thread = threading.Thread(target=self.key_listener, args=(rec_ev_hl,))
        listener_thread.start()

        # Main Loop
        cv.imshow(self.__wind_name, self.__md.up_rgb_map)
        while True:
            rec_ev_hl.update_image()
            cv.imshow(self.__wind_name, rec_ev_hl.canvas)

            c2 = cv.getWindowProperty(self.__wind_name, cv.WND_PROP_VISIBLE) < 1
            if c2 or (cv.waitKey(20) & 0xFF == 13 or cv.waitKey(20) & 0xFF == 27):
                break
        #
        listener_thread.join()
        cv.destroyAllWindows()

    def init_wind(self):
        cv.namedWindow(self.__wind_name, cv.WINDOW_GUI_NORMAL)

    def key_listener(self, handler: RecEventHandler):
        pass
        # todo
        # with Listener(on_press=handler.on_key_press, on_release=handler.on_key_release) as listener:
        #     listener.join()

    # def get_district_vis(self) -> VisualRectangle:
    #     if self.__cfg.whole_map:
    #         print(f"[{self.__ns}]: whole map is selected as district for CPP! There is no rec_ev_hl!")
    #         return None
    #     return self.__rec_ev_hl.get_main_obj()

    def get_district_rect(self) -> Rectangle:
        if not self.__cfg.whole_map:
            return self.__rec_ev_hl.get_main_obj().get_rect()

        # create district based on map data
        rot = 0
        xc = self.__md.up_c_w
        yc = self.__md.up_c_h
        w = self.__md.updated_w
        h = self.__md.updated_h
        data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        return Rectangle(data)
