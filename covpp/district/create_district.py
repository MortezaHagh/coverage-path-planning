"""District selection for coverage path planning (GUI or whole-map).

Steps:
- If ``PipeLineConfig.whole_map`` is True, create a rectangle covering the map
- Otherwise open a window and let the user draw/select the district rectangle

Result:
- A `Rectangle` stored in `DistrictGen.rectangle`.
"""
import threading
import cv2 as cv
from pynput.keyboard import Listener
from covpp.base.logger import MyLogger
from covpp.base.map_data import MapData
from covpp.base.configs import PipeLineConfig
from covpp.gui.rect_event_hl import RecEventHandler
from covpp.gui.rectangle import Rectangle


class DistrictGen:
    """Create a district rectangle via GUI or whole-map fallback.

    Args:
    - map_data: The `MapData` used to infer dimensions and display imagery.

    Attributes:
    - rectangle: The created district rectangle.
    """

    def __init__(self, map_data: MapData):
        self.lg = MyLogger("DistrictEvent")
        self.__wind_name: str = "Select District"
        self.lg.info1("Step: creating district for CovPP")

        # attributes
        self.__md: MapData = map_data
        self.__cfg = PipeLineConfig()

        # resulted rectangle data
        self.rectangle: Rectangle = None

        # check if whole map is selected or should use GUI
        if self.__cfg.whole_map:
            self.lg.info2("whole map is selected as district for CPP!")
            self.create_district_whole()
            return None

        self.lg.info2("Selecting district for CPP by GUI interaction!")
        self.create_district_gui()

        # # evaluate rectangle # later
        # self.validate_district_rectangle()

    def create_district_whole(self):
        """Create a full-map district rectangle based on `MapData` dimensions."""
        rot = 0
        xc = self.__md.up_c_w
        yc = self.__md.up_c_h
        w = self.__md.updated_w
        h = self.__md.updated_h
        rect_data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self.rectangle = Rectangle(rect_data)

    def create_district_gui(self) -> None:
        """Create a district rectangle via interactive GUI selection."""
        # init window
        self.init_wind()

        # init rectangle event handler
        rec_ev_hl = RecEventHandler(self.__md.up_rgb_map)

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

        # get rectangle
        self.rectangle = rec_ev_hl.get_main_obj().get_rect()

    def init_wind(self):
        cv.namedWindow(self.__wind_name, cv.WINDOW_GUI_NORMAL)

    def key_listener(self, handler: RecEventHandler):
        with Listener(on_press=handler.on_key_press, on_release=handler.on_key_release) as listener:
            listener.join()

    def get_district_rect(self) -> Rectangle:
        """Return the created district rectangle."""
        return self.rectangle
