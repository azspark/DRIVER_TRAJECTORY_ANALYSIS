import numpy as np 
import pandas as pd 
from scipy.stats import describe

def porto_timestamps(start_t, length):
    # print(start_t, length)
    return np.array([start_t + i * 15 for i in range(length)])


class Engrider:
    def __init__(self, geo_range, grid_lon_div, grid_lat_div):
        self.geo_range = geo_range
        self.grid_lon_div = grid_lon_div
        self.grid_lat_div = grid_lat_div
        self._calculate_unit()

    def _calculate_unit(self):
        self.lon_unit = (self.geo_range['lon_max'] - self.geo_range['lon_min']) / self.grid_lon_div
        self.lat_unit = (self.geo_range['lat_max'] - self.geo_range['lat_min']) / self.grid_lat_div

    def to_grid_id(self, df):
        region_lat_pos = int((df[1] - self.geo_range['lat_min']) / self.lat_unit)
        region_lon_pos = int((df[0] - self.geo_range['lon_min']) / self.lon_unit)
        return region_lon_pos, region_lat_pos


def base_stat(data, count_pos_neg=False):
    """Calculate basic statistics for sequence of data
    
    - count_pos_neg: seperately count statistics for positive and negetive part of data
    """
    if count_pos_neg:
        pos_idx = data > 0.0
        return base_stat(data[pos_idx], False) + base_stat(data[~pos_idx], False)
    if len(data) > 0:
        return [data.mean(), data.var()] + list(np.percentile(data, [10,50,90]))
    else:
        return [0.0 for i in range(5)]