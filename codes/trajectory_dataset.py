from trajectory import Trajectory
from driver import Driver
from city_cells import CityCells
import numpy as np 
from utils import *

class TrajectoryDataset:
    """Manipulate trajectories and cells information, prepare data for feature engineering""" 
    def __init__(self, df, count_cell_info=False, data_type='porto', geo_range=None, div_number=None, turning_threshold=0.3):
        """
        Args:
        - df: pd.DataFrame, at least include following information with specified column names:
            ['LABEL', 'TIMESTAMP', 'DISTANCE', 'SD', 'TAXI_ID', 'POLYLINE']
        - count_cell_info: boolean, if true will add feature of each region cells.
        - data_type: data_type=porto means input timestamp is a single value each point's sampling rate is 15s
          otherise, input will be a list of timestamps
        """
        self.df = df
        self.with_cell_info = count_cell_info
        self.turning_threshold = turning_threshold
        self._init_trajectories()
        if count_cell_info:
            self.city_cells = CityCells(self.trajectories, geo_range, div_number)

    def _dataframe_preprocess(self):
        """Generate more feature for analysis"""

        # 2. group df for specific driver analysis
        self.grouped_df = self.df.groupby('TAXI_ID')

    def _init_trajectories(self):
        """Initilize trajectory objects"""
        
        self.trajectories = [Trajectory(r['POLYLINE'], porto_timestamps(r['TIMESTAMP'], len(r['POLYLINE'])), r['LABEL'], turning_threshold=self.turning_threshold) for i, r in self.df.iterrows()]
        
    
    def generate_trajectories_feature(self):
        """Generate data for feature engineering

        Returns:
        - df_traj_feature: pd.DataFrame
        - label: trajectory driver labels
        """
        trajs_feature = [traj.get_basic_feature() for traj in self.trajectories]
        self.df_feature = pd.DataFrame(trajs_feature)
        self.df_feature["LABEL"] = self.df["LABEL"]

    
    def driver_statistics(self):
        """Overall statistics of every driver
        
        Return:
        - driver_df: pd.DataFrame
        """
        pass

    def get_driver(self, driver_id):
        """Get Driver class of specified id for further analysis"""
        
        # TODO add df_feature information
        return Driver(self.trajectories, self.grouped_df.get_group(driver_id), None)