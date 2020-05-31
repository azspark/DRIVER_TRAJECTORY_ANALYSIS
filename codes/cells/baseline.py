import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl
import time
from multiprocessing import Process, Lock
from sklearn.metrics import f1_score
from pathlib import Path
import logging
from .point import Point
from .cell import Cell
from .city_cells import CityCells

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
    raw_data_path = '/data3/zhuzheng/trajectory/porto/porto_1.0h.pkl'
    # raw_data_path = 'data/porto_1000.pkl' # for debug
    with open(raw_data_path, 'rb') as f:
        df = pkl.load(f)
    df['TRAJ_ID'] = list(range(len(df)))
    df['TRAJ_ID'] = df['TRAJ_ID'].astype(np.int)

    pair_path = '/data3/zhuzheng/trajectory/mcan/porto/1.0h/'
    train = pd.read_csv(pair_path + 'train.csv', usecols=[1,2])
    test = pd.read_csv(pair_path + 'test.csv', usecols=[1,2])
    train_idx = list(set(train['ltable_id'].tolist()).union(train['rtable_id'].tolist()))

    # train_idx = list(range(80))
    # test = pd.DataFrame({'ltable_id':pd.Series([92,93,94]), 
    #                     'rtable_id':pd.Series([97,91,98])})

    train_df = df.reindex(train_idx)
    

    city_cells = CityCells(cell_num)
    city_cells.add_trips(train_df)
    print(f'done: {cell_num}, {k}')

    total = len(test)
    y_true = list()
    y_pred = list()
    # import pdb; pdb.set_trace()
    for _, row in test.iterrows():
        pair1, pair2 = row['ltable_id'], row['rtable_id']
        state = df.loc[pair1]['TAXI_ID'] == df.loc[pair2]['TAXI_ID']
        pred = city_cells.judge(df.loc[pair1]['POLYLINE'], df.loc[pair2]['POLYLINE'], k)
        y_true.append(state)
        y_pred.append(pred)
    f1_binary = f1_score(y_true, y_pred, average='binary')
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    # f1_micro = f1_score(y_true, y_pred, average='micro')
    # f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f1_binary)

    try:
        lock.acquire()
        logger.info(f'cell_num = {cell_num:0>3d}, k = {k:0>2d}, {f1_binary:.5f}')

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

    lock = Lock()
    one_process(50, 15, lock)
    p_obj = []
    for i in range(2, 6):  # num cell
        for j in range(5, 9):  # top-k
            p = Process(target=one_process, args=(i * 10, j * 2, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    # p_obj = []
    # for i in range(8, 13):  # num cell
    #     for j in range(5, 9):  # top-k
    #         p = Process(target=one_process, args=(i * 3, j * 5, lock))
    #         p_obj.append(p)
    #     print('Waiting for all subprocesses done...')

    # for i in p_obj:
    #     i.start()
    # for i in p_obj:
    #     i.join()

    
    # p_obj = []
    # for i in range(11, 15):  # num cell
    #     for j in range(5, 9):  # top-k
    #         p = Process(target=one_process, args=(i * 6, j * 10, lock))
    #         p_obj.append(p)
    #     print('Waiting for all subprocesses done...')

    # for i in p_obj:
    #     i.start()
    # for i in p_obj:
    #     i.join()

    # p_obj = []
    # for i in range(2, 6):  # num cell
    #     for j in range(5, 9):  # top-k
    #         p = Process(target=one_process, args=(i * 7, j * 4, lock))
    #         p_obj.append(p)
    #     print('Waiting for all subprocesses done...')

    # for i in p_obj:
    #     i.start()
    # for i in p_obj:
    #     i.join()

    # p_obj = []
    # for i in range(15, 20):  # num cell
    #     for j in range(5, 9):  # top-k
    #         p = Process(target=one_process, args=(i * 3, j * 2, lock))
    #         p_obj.append(p)
    #     print('Waiting for all subprocesses done...')

    # for i in p_obj:
    #     i.start()
    # for i in p_obj:
    #     i.join()