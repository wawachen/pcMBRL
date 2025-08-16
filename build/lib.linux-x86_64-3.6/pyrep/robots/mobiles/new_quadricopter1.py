from pyrep.robots.mobiles.new_drone_base1 import NewDrone_base


class NewQuadricopter(NewDrone_base):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')