import pandas as pd 
from geopy.distance import distance
import numpy as np 

def stay_times_func(mins):
    def stay_times(df):
        times = df['TIMESTAMPS']
        coords = df['POLYLINE']
        gaps = times[1:] - times[:-1]
        is_gap = gaps > mins * 60
        return np.sum(is_gap), np.sum(gaps[is_gap]) / 3600, get_mean_distance(coords, is_gap)
    return stay_times

def get_mean_distance(coords, is_gap):
    distances = 0.0
    count  = 0.0
    
    for idx in np.argwhere(is_gap):
        idx = int(idx)
        p1 = (coords[idx][1], coords[idx][0])
        p2 = (coords[idx+1][1], coords[idx+1][0])
        distances += distance(p1, p2).meters
        count += 1
    if count == 0.0:
        return 0.0
    else:
        return distances / count