import folium
import numpy as np 
import datetime
from .vis_obj import Viser

class Driver(Viser):
    """Record and analyse driving behaviors of one driver"""
    
    def __init__(self, trajectories, df_driver, df_features):
        Viser.__init__(self, trajectories, df_driver)
        self.df_features = df_features

