""" Create district for coverage path planning. GUI version: select rectangle area as district. """
import threading
import cv2 as cv
from pynput.keyboard import Listener
from scripts.base.logger import MyLogger
from scripts.base.map_data import MapData
from scripts.base.configs import PipeLineConfig
from scripts.gui.rect_event_hl import RecEventHandler
from scripts.gui.rectangle import Rectangle


class DistrictGen:
    """ Create district for coverage path planning. GUI version: select rectangle area as district. """

    def __init__(self, map_data: MapData):
        self.lg = MyLogger("DistrictEvent")
        self.__wind_name: str = "Select District"
        self.lg.info1("Step: creating district for CovPP")

        # attributes
        self.__md: MapData = map_data
        self.__cfg: PipeLineConfig = PipeLineConfig()

        # check if whole map is selected
        if self.__cfg.whole_map:
            self.lg.info2("whole map is selected as district for CPP!")
            return None
        self.lg.info2("Selecting district for CPP ... ")

        # init window
        self.init_wind()

        # init rectangle event handler
        rec_ev_hl = RecEventHandler(map_data.up_rgb_map)
        self.__rec_ev_hl: RecEventHandler = rec_ev_hl

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
        with Listener(on_press=handler.on_key_press, on_release=handler.on_key_release) as listener:
            listener.join()

    # def get_district_vis(self) -> VisualRectangle:
    #     if self.__cfg.whole_map:
    #         self.lg.info(f"[{self.__ns}]: whole map is selected as district for CPP! There is no rec_ev_hl!")
    #         return None
    #     return self.__rec_ev_hl.get_main_obj()

    def get_district_rect(self) -> Rectangle:
        """ Get district rectangle based on the selected rectangles in the GUI or whole map."""

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
