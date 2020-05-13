

class Driver:
    """Record and analysis of driving behaviors from one driver"""
    
    def __init__(self, trajectories, df_driver, df_features):
        self.trajectories = trajectories
    
    def start_point_distribution_visulization(self):
        pass

    def vis_trajs(self, shown_number, time_interval=None):
        """Visualization of driver trajectories which filtered by some rules"""