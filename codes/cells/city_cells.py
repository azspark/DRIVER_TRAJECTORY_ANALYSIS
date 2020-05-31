import os
import math
import random
import numpy as np
import pandas as pd
from collections import Counter
import h5py
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm
from multiprocessing import Process, Lock
import sys
import os
from pathlib import Path
import logging
from .cell import Cell
from .point import Point

# geo_range = {'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}

class CityCells:
    def __init__(self, div_number=100, geo_range={'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}):
        """

        :param div_number:
        :param geo_range:
        """
        self.point2cell = dict()
        self.div_number = div_number
        self.min_x, self.min_y, self.max_x, self.max_y = geo_range['lon_min'], geo_range['lat_min'], geo_range['lon_max'], geo_range['lat_max']
        self.gap_x, self.gap_y = self._cell_gap(self.div_number)

    def _cell_gap(self, nums=100):
        """find the gap for the target nums

        Args:
            nums (int, optional): the cell number of one side. Defaults to 100.
        """
        gap_x = (self.max_x - self.min_x) / nums
        gap_y = (self.max_y - self.min_y) / nums
        return gap_x, gap_y
    
    def _add_trip(self, driver_id, traj_id, trip):
        """
        add one triajectory to the city cells
        :param driver_id: (int)
        :param traj_id: (int)
        :param trip: (np.array) POLYLINE
        :return:
        """
        x = np.floor((trip[:, 0] - self.min_x) / self.gap_x)
        x = x.astype(np.int)
        y = np.floor((trip[:, 1] - self.min_y) / self.gap_y)
        y = y.astype(np.int)

        assert len(x) == len(y)
        for i in range(len(x)):
            point = Point(x[i], y[i])
            if point not in self.point2cell:
                self.point2cell[point] = Cell(point)
            self.point2cell[point].add(driver_id, traj_id)

    def add_trips(self, df):
        """add the train trajectories to the cells

        :param df: pd.DataFrame, ['LABEL', 'TIMESTAMP', 'DISTANCE', 'SD', 'TAXI_ID', 'POLYLINE', 'TRAJ_ID']
        :return: None
        """
        for idx, traj in df.iterrows():
            traj_id = traj['TRAJ_ID'] # id of the trajectory
            driver_id = traj['TAXI_ID']
            trip = np.array(traj['POLYLINE'])
            if sum(trip[:, 0] < self.min_x) > 0 or sum(trip[:, 0] > self.max_x) > 0:
                continue
            if sum(trip[:, 1] < self.min_y) > 0 or sum(trip[:, 1] > self.max_y) > 0:
                continue
            self._add_trip(driver_id, traj_id, trip)



    def _find_topk_driver(self, trip, k=10):
        """

        :param trip: np.array
        :param k:
        :return: driver_ids: list [driver1, driver2, driver3]
        """
        x = (np.floor((trip[:, 0] - self.min_x) / self.gap_x)).astype(np.int)
        y = (np.floor((trip[:, 1] - self.min_y) / self.gap_y)).astype(np.int)

        ans = Counter(dict())
        assert len(x) == len(y)
        for i in range(len(x)):
            point = Point(x[i], y[i])
            # pay attention to the following lines, these are algorithms for top-k
            if point in self.point2cell:
                tmp = Counter(dict(self.point2cell[point].topk_drivers(k)))
                ans = ans + tmp

        ans = dict(ans)
        ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
        k = min(k, len(ans))
        drivers = [ans[i][0] for i in range(k)]
        return drivers

    def judge(self, trip1, trip2, k=10):
        """

        :param trip1: [[longi, lati]]
        :param trip2: [[longi, lati]]
        :return: bool whether match
        """
        trip1 = np.array(trip1)
        trip2 = np.array(trip2)
        drivers1 = self._find_topk_driver(trip1, k)
        drivers2 = self._find_topk_driver(trip2, k)        
        
        theshold = k/2 if k < 20 else math.sqrt(k)/1.3
        if len(set(drivers1).intersection(set(drivers2))) > theshold:
            return True
        else:
            return False


    def __str__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f

    def __repr__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f