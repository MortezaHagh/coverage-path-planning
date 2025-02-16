from scripts.base.configs import SpaceConfigs  # BConfigs
from scripts.district.district import District


class WF:
    def __init__(self, district: District):
        self.wf_dist = SpaceConfigs.wf_r_dist
        self.district = district
