import numpy as np 


class Cell:
    """Cell represents a rectangle region on map
    
    Records some drving information of passed trajectories.
    """
    def __init__(self, idx, x_y):
        self.idx = idx
        self.x_y = x_y

        self.speeds = self._init_data_dict()
        self.steer_speeds = self._init_data_dict()

    def _init_data_dict(self):
        return dict([(i, []) for i in range(24)])

    def update_speeds(self, hour_time, speeds):
        self.speeds[hour_time] += speeds
    
    def update_steer_speeds(self, hour_time, speeds):
        self.steer_speeds[hour_time] += speeds

    def calculate_mean_var(self):
        self.speed_stat = self._calculate_mean_var(self.speeds)
        self.steer_speed_stat = self._calculate_mean_var(self.steer_speeds)

    def _calculate_mean_var(self, speed_dict):
        out = []
        for i in range(24):
            vals = speed_dict[i]
            out.append([np.mean(vals), np.var(vals)])
        return out