from pyrep.robots.mobiles.RL_drone_base import RLDrone_base
from pyrep.robots.mobiles.RL_drone_base_withoutCamera import RLDrone_base_nc
from pyrep.robots.mobiles.RL_drone_base_withoutCamera1 import RLDrone_base_nc1
from pyrep.robots.mobiles.RL_drone_base_withCamera import RLDrone_base_wc


class RLQuadricopter(RLDrone_base):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')

class RLQuadricopter_nc(RLDrone_base_nc):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')

class RLQuadricopter_nc1(RLDrone_base_nc1):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')

class RLQuadricopter_wc(RLDrone_base_wc):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')
