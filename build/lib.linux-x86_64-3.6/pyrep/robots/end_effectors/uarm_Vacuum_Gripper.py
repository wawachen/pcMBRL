from pyrep.robots.end_effectors.suction_cup import SuctionCup


class UarmVacuumGripper(SuctionCup):

    def __init__(self, count: int = 0):
        super().__init__(count, 'uarmVacuumGripper')
