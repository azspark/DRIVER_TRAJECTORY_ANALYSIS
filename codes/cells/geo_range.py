import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl
import time
from multiprocessing import Process, Lock
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import logging
from tqdm import tqdm
from .point import Point
from .cell import Cell
from .city_cells import CityCells
from .config import setting

def run():
    minx, miny, maxx, maxy = 1e8, 1e8, 1e-8, 1e-8

    raw_data_path = setting['raw_data_path']
    with open(raw_data_path, 'rb') as f:
        df = pkl.load(f)
    df['TRAJ_ID'] = list(range(len(df)))
    df['TRAJ_ID'] = df['TRAJ_ID'].astype(np.int)
    for idx, traj in tqdm(df.iterrows(), desc=""):
        trip = np.array(traj['POLYLINE'])
        # import pdb; pdb.set_trace()
        tminx = np.min(trip[:, 0])
        if tminx > 101:
            minx = min(minx, tminx)
        tmaxx = np.max(trip[:, 0])
        if tmaxx < 105:
            maxx = max(maxx, tmaxx)
        tminy = np.min(trip[:, 1])
        if tminy > 1.1:
            miny = min(miny, tminy)
        tmaxy = np.max(trip[:, 1])
        if tmaxy < 1.5:
            maxy = max(maxy, tmaxy)
        # minx = min(minx, np.min(trip[:, 0]))
        # maxx = max(maxx, np.max(trip[:, 0]))
        # miny = min(miny, np.min(trip[:, 1]))
        # maxy = max(maxy, np.max(trip[:, 1]))

    print(minx)
    print(maxx)
    print(miny)
    print(maxy)

if __name__ == "__main__":
    run()