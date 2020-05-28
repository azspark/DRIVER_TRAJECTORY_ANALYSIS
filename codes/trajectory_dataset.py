from .trajectory import Trajectory
from .driver import Driver
from .city_cells import CityCells
from .OD import OD
import numpy as np 
from .utils import *

class TrajectoryDataset:
    """Manipulate trajectories and cells information, prepare data for feature engineering""" 
    def __init__(self, df, count_cell_info=False, data_type='porto', timezone='Europe/Lisbon',
         use_graph_feature=False, geo_range=None, div_number=None, turning_threshold=0.3, 
         acc_threshold=0.1, count_od_info=True, base_feat=True):
        """
        Args:
        - df: pd.DataFrame, at least include following information with specified column names:
            ['LABEL', 'TIMESTAMP', 'DISTANCE', 'SD', 'TAXI_ID', 'POLYLINE']
        - count_cell_info: boolean, if true will add feature of each region cells.
        - data_type: data_type=porto means input timestamp is a single value each point's sampling rate is 15s
          otherise, input will be a list of timestamps
        - use_graph_feature: boolean, if true we will calculate driving state trainstion graph
        - geo_range: dict, store longtitude and latitude range for spliting cells
        - div_number: int, city will be splited into div_number * div_number cells
        - turning_threshold: float, angle threshold to determinte wheather a driver is steering
        - acc_threshold: float, accleration threshold 
        - count_od_info: boolean, with Origin Destination information, if not add it
        - base_feat: boolean, if calculate basic feature of each trajectory such as speed, accleration
        """
        self.df = df
        self.df.set_index(np.arange(df.shape[0]), inplace=True)
        self.df['LABEL'] = pd.factorize(df['TAXI_ID'])[0].astype(np.uint16)
        self.with_cell_info = count_cell_info
        self.turning_threshold = turning_threshold
        self.use_graph_feature = use_graph_feature
        self.acc_threshold = acc_threshold
        self.timezone = timezone
        self.data_type = data_type
        self.count_od_info = count_od_info
        self.base_feat = base_feat

        self._dataframe_preprocess()
        self._init_trajectories()
        if count_cell_info:
            self.city_cells = CityCells(self.trajectories, geo_range, div_number)

        self.df_feature = None

    def _dataframe_preprocess(self):
        """Generate more feature for analysis"""
        # 1. add baisc feature like date, time in day, ....
        if self.data_type != 'porto':
            self.df['TIMESTAMP'] = self.df.apply(lambda df: df['TIMESTAMPS'][0], axis=1)
        self.df['TIME'] = pd.to_datetime(self.df['TIMESTAMP'], unit='s', utc=True)
        
        self.df.TIME = self.df.TIME.dt.tz_convert(self.timezone)
        # 2. group df for specific driver analysis
        self.grouped_df = self.df.groupby('LABEL')
        if self.count_od_info:
            if 'SD' not in self.df.columns:
                self._add_OD_info()
            self.grouped_od = self.df.groupby('SD')

    def _add_OD_info(self):
        pass

    def _init_trajectories(self):
        """Initilize trajectory objects"""
        if self.data_type == 'porto':
            self.trajectories = [Trajectory(r['POLYLINE'], porto_timestamps(r['TIMESTAMP'], len(r['POLYLINE'])), r['LABEL'], traj_id=i, base_feat=self.base_feat,
                turning_threshold=self.turning_threshold, use_graph_feature=self.use_graph_feature, acc_threshold=self.acc_threshold) for i, r in self.df.iterrows()]
        else:
            self.trajectories = [Trajectory(r['POLYLINE'], r['TIMESTAMPS'], r['LABEL'], traj_id=i, base_feat=self.base_feat,
                turning_threshold=self.turning_threshold, use_graph_feature=self.use_graph_feature, acc_threshold=self.acc_threshold) for i, r in self.df.iterrows()]
    
    def generate_trajectories_feature(self):
        """Generate data for feature engineering

        Returns:
        - df_traj_feature: pd.DataFrame
        - label: trajectory driver labels
        """
        if self.df_feature is not None:
            return self.df_feature
        trajs_feature = [traj.get_basic_feature() for traj in self.trajectories]
        self.df_feature = pd.DataFrame(trajs_feature)
        self.df_feature["LABEL"] = self.df["LABEL"]
        return self.df_feature

    def generate_graph_feature(self):
        """Generate drving state trainsition graph feature"""
        traj_graph_feature = [traj.get_graph_feature() for traj in self.trajectories]
        self.df_graph_feature = pd.DataFrame(traj_graph_feature)
        self.df_graph_feature["LABEL"] = self.df["LABEL"]
        return self.df_graph_feature

    
    def driver_statistics(self):
        """Overall statistics of every driver
        
        Return:
        - pd.DataFrame
        """
        df_feature_ = self.df_feature[~self.df_feature[0].values][['LABEL', 1, 2, 6, 7, 11, 12, 16, 17, 21, 22, 26, 27]]
        df_feature_.columns = ['LABEL', 'Speed_Mean', 'Speed_Var', "Ac_Mean", "Ac_Var", "Dc_Mean", "Dc_Var", 
        'Steer_Speed_Mean', 'Steer_Speed_Var', "Steer_Ac_Mean", "Steer_Ac_Var", "Steer_Dc_Mean", "Steer_Dc_var"]
        df_feature_['DISTANCE'] = self.df[~self.df_feature[0].values]['DISTANCE']
        return df_feature_.groupby('LABEL').mean()

    def get_driver(self, driver_id):
        """Get Driver class of specified id for further analysis"""
        
        # TODO add df_feature information
        return Driver(self.trajectories, self.grouped_df.get_group(driver_id), None)

    def od_num_info(self):
        return self.grouped_od.size().sort_values(ascending=False)

    def get_od(self, od_id):
        return OD(self.trajectories, self.grouped_od.get_group(od_id))