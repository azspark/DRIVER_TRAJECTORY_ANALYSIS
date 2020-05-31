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

# geo_range = {'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/baseline.log')
file_handler.setLevel(level=logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




def one_process(cell_num, k, lock):
    """process one line

    Args:
        num_cell ([type]): divide the city into num_cell*num_cell cells
        k ([type]): top-k
        lock ([type]): [description]
    """
    raw_data_path = setting['raw_data_path']
    # raw_data_path = 'data/porto_1000.pkl' # for debug
    
    with open(raw_data_path, 'rb') as f:
        df = pkl.load(f)
    df['TRAJ_ID'] = list(range(len(df)))
    df['TRAJ_ID'] = df['TRAJ_ID'].astype(np.int)

    pair_path = setting['pair_path']
    train = pd.read_csv(pair_path + 'train.csv', usecols=[1,2])
    test = pd.read_csv(pair_path + 'test.csv', usecols=[1,2])
    train_idx = list(set(train['ltable_id'].tolist()).union(train['rtable_id'].tolist()))

    # test = test.reindex(np.unique(np.random.choice(len(test), 1100)))
    #
    # train_idx = list(range(80))
    # test = pd.DataFrame({'ltable_id':pd.Series([92,93,94,95,93]), 
    #                     'rtable_id':pd.Series([97,91,98,98,90])})

    train_df = df.reindex(train_idx)

    pair1 = df.reindex(test['ltable_id'])
    pair2 = df.reindex(test['rtable_id'])
    assert len(pair1) == len(pair2)

    del df
    import gc
    gc.collect()

    # add train trajectories
    geo_range = setting['geo_range']
    city_cells = CityCells(cell_num, geo_range)
    city_cells.add_trips(train_df)
    print(f'done: {cell_num}, {k}')


    y_true = list()
    y_pred = list()

    for (_,row1), (_,row2) in tqdm(zip(pair1.iterrows(), pair2.iterrows()), desc=""):
        state = row1['TAXI_ID'] == row2['TAXI_ID']
        pred = city_cells.judge(row1['POLYLINE'], row2['POLYLINE'], k)
        y_true.append(state)
        y_pred.append(pred)
    y_pred = np.array(y_pred).astype(np.int)
    y_true = np.array(y_true).astype(np.int)

    f1_binary = f1_score(y_true, y_pred, average='binary')

    # import pdb; pdb.set_trace()
    precision, recall, f1, _  = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])

    print(sum(y_true), sum(y_pred))
    print(precision, recall, f1)
    print(f1_binary)

    try:
        lock.acquire()
        logger.info(f'cell_num = {cell_num:0>3d}, k = {k:0>2d}, {f1_binary:.5f}')
        logger.info(f'{sum(y_true)}, {sum(y_pred)},{precision}, {recall}, {f1}')

    except Exception as e:
        print(e)
    finally:
        lock.release()


if __name__ == '__main__':
    """usage:
        python -m codes.cells.baseline
    """
    # os.chdir('/home/yangyueren/code/DRIVER_TRAJECTORY_ANALYSIS')
    print(os.getcwd())
    random.seed(10086)
    np.random.seed(10086)

    lock = Lock()
    # one_process(24, 18, lock)
    p_obj = []
    for i in range(2, 4):  # num cell
        for j in range(5, 7):  # top-k
            p = Process(target=one_process, args=(i * 10, j * 2, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    for i in range(8, 10):  # num cell
        for j in range(6, 8):  # top-k
            p = Process(target=one_process, args=(i * 3, j * 3, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    
    p_obj = []
    for i in range(10, 13):  # num cell
        for j in range(5, 8):  # top-k
            p = Process(target=one_process, args=(i * 10, j * 3, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    for i in range(2, 5):  # num cell
        for j in range(4, 7):  # top-k
            p = Process(target=one_process, args=(i * 7, j * 1, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    for i in range(15, 18):  # num cell
        for j in range(5, 8):  # top-k
            p = Process(target=one_process, args=(i * 3, j * 2, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()