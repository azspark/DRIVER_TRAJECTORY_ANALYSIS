from .cell import Cell
from .utils import Engrider

class CityCells:
    """Index, control and update the cell information"""
    def __init__(self, trajectories, geo_range=None, div_number=None):
        
        self.update_info(trajectories)
    
    def update_info(self, trajectories):
        pass

    def vis_city_info(self):
        pass