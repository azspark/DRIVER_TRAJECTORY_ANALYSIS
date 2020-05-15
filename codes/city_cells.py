from .cell import Cell
from .utils import Engrider
import numpy as np 
import matplotlib.pyplot as plt 

class CityCells:
    """Index, control and update the cell information"""
    def __init__(self, trajectories, geo_range=None, div_number=None):
        self.update_info(trajectories)
    
    def _init(self):
        pass


    def update_info(self, trajectories):
        
        for traj in trajectories:
            pass

    def vis_city_info(self):
        pass