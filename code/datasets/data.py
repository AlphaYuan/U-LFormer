from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.datasets import load_from_tsfile_to_dataframe





#*************************************************************************************************************

"""
Code to load Track datasets. 
"""

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


#*************************************************************************************************************


logger = logging.getLogger('__main__')




class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class Trackdata(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
    """

    def __init__(self, track=None, dim=3, n_proc=1, limit_size=None, config=None):
        id = [i for i in range(len(track))]
        track_id_len = []  # 记录每一个轨迹的长度
        iindex = []  # 最终dataframe的索引
        columns=[]
        columns_original = []
        total_len = 0  # dataframe有多少行
        #label_a = []  # 回归任务标签
        #label_model = []  # 分类任务标签

        track_diff = [[[] for _ in range(len(track))]for _ in range(dim)]
        track_diff_original = [[[] for _ in range(len(track))]for _ in range(dim)]

        for i in range(dim):
            columns.append('dim_{}'.format(i))
        for i in range(2):
            columns_original.append('dim_{}'.format(i))

        # columns = ['dim_0', 'dim_1']  # 最终dataframe的columns
        trackx = [[] for _ in range(len(track))]
        tracky = [[] for _ in range(len(track))]

        columns_label = ['a', 'K', 'state']  # 最终dataframe的columns
        track_alpha = [[] for _ in range(len(track))]
        track_K = [[] for _ in range(len(track))]
        track_state = [[] for _ in range(len(track))]

        log_func = math.log10
        if 'log' in config:
            log_func = math.log
        print(log_func)

        for i in range(len(track)):
            noise_x = 0
            noise_y = 0
            if 'add_noise' in config and config['add_noise'] == True:
                if ('test_pattern' in config and config['test_pattern'] == True):
                    t=1
                else:
                    # print('\nadd_noise\n')
                    noise_x = random.gauss(0, 0.5)
                    noise_y = random.gauss(0, 0.5)
            tracklen = int((len(track[i])) / (2 + 3))   #2+3是数据一共有五个元素  x，y，a，k，state
            track_id_len.append(tracklen)  # 第i条轨迹的长度
            resultx = track[i][0:tracklen]
            trackx[i] = [p - resultx[0] + noise_x for p in resultx]  # 减去第一个元素
            resulty = track[i][tracklen:2 * tracklen]
            tracky[i] = [p - resulty[0] + noise_y for p in resulty]

            track_alpha[i] = track[i][2 * tracklen:3 * tracklen]
            resultK = track[i][3 * tracklen:4 * tracklen]
            track_K[i] = [log_func(p) if p > 9e-13 else log_func(1e-13) for p in resultK] # 针对immobile中的K=0处理
            #track_K[i] =track[i][3 * tracklen:4 * tracklen]
            track_state[i] = track[i][4 * tracklen:5 * tracklen]



            # label_a.append(track[i][1])
            # label_model.append(track[i][0])
            for j in range(tracklen):
                iindex.append(i)

        trackx_flip = [row[::-1] for row in trackx]
        tracky_flip = [row[::-1] for row in tracky]
        trackx_flip_ = [i[1:-1] for i in trackx_flip]
        tracky_flip_ = [i[1:-1] for i in tracky_flip]



        trackx_new = [x + y + z for x, y, z in zip(trackx, trackx_flip_, trackx)]
        tracky_new = [x + y + z for x, y, z in zip(tracky, tracky_flip_, tracky)]

        tracktrans_original = np.zeros((len(iindex), 2))
        tracktrans = np.zeros((len(iindex), dim))
        tracktrans_label = np.zeros((len(iindex), 3))

        track_alpha = np.asarray(track_alpha, dtype=object)
        track_K = np.asarray(track_K, dtype=object)
        track_state = np.asarray(track_state, dtype=object)

        for i in range(0,dim,2):
            for j in range(len(track)):
                traj_x=trackx_new[j]
                traj_y=tracky_new[j]
                for t in range(track_id_len[j]):
                    tmpx = traj_x[t + int(i / 2) + 1] - traj_x[t]
                    tmpy = traj_y[t + int(i / 2) + 1] - traj_y[t]
                    track_diff[i][j].append(tmpx)
                    track_diff[i+1][j].append(tmpy)
            #track_diff[i] = np.asarray(track_diff[i])
            #track_diff[i+1] = np.asarray(track_diff[i+1])


        for d in range(dim):
            t = 0
            for i in range(len(track)):
                for j in range(track_id_len[i]):
                    tracktrans[t + j][d] = track_diff[d][i][j]
                t = t + track_id_len[i]

        # track original 
        for i in range(0,2,2):
            for j in range(len(track)):
                traj_x=trackx[j]
                traj_y=tracky[j]
                for t in range(track_id_len[j]):
                    # tmpx = traj_x[t + int(i / 2) + 1] - traj_x[t]
                    # tmpy = traj_y[t + int(i / 2) + 1] - traj_y[t]
                    track_diff_original[i][j].append(traj_x[t])
                    track_diff_original[i+1][j].append(traj_y[t])
            #track_diff[i] = np.asarray(track_diff[i])
            #track_diff[i+1] = np.asarray(track_diff[i+1])


        for d in range(2):
            t = 0
            for i in range(len(track)):
                for j in range(track_id_len[i]):
                    tracktrans_original[t + j][d] = track_diff_original[d][i][j]
                t = t + track_id_len[i]        
        # t = 0
        # for i in range(len(track)):
        #     for j in range(track_id_len[i]):
        #         tracktrans[t + j][0] = trackx_new[i][j]
        #     t = t + track_id_len[i]

        # t = 0
        # for i in range(len(track)):
        #     for j in range(track_id_len[i]):
        #         tracktrans[t + j][1] = tracky_new[i][j]
        #     t = t + track_id_len[i]

        t = 0
        for i in range(len(track)):
            for j in range(track_id_len[i]):
                tracktrans_label[t + j][0] = track_alpha[i][j]
            t = t + track_id_len[i]

        t = 0
        for i in range(len(track)):
            for j in range(track_id_len[i]):
                tracktrans_label[t + j][1] = track_K[i][j]
            t = t + track_id_len[i]
        
        t = 0
        for i in range(len(track)):
            for j in range(track_id_len[i]):
                tracktrans_label[t + j][2] = track_state[i][j]
            t = t + track_id_len[i]

        self.track_id_len = track_id_len

        df = pd.DataFrame(data=tracktrans, index=iindex, columns=columns)
        self.all_IDs = df.index.unique()
        self.all_df = df
        self.feature_names = df.columns
        self.feature_df = self.all_df
        self.max_seq_len = max(track_id_len)
        self.config = config

        df_label = pd.DataFrame(data=tracktrans_label, index=iindex, columns=columns_label)
        self.all_IDs_label = df_label.index.unique()
        self.all_df_label = df_label
        self.feature_names_label = df_label.columns
        self.feature_df_label = self.all_df_label
        self.max_seq_len_label = max(track_id_len)
        self.config = config

        df_original = pd.DataFrame(data=tracktrans_original, index=iindex, columns=columns_original)
        self.all_IDs_original = df_original.index.unique()
        self.all_df_original = df_original
        self.feature_names_original = df_original.columns
        self.feature_df_original = self.all_df_original
        self.max_seq_len_original = max(track_id_len)
        self.config = config

        self.labels_r = df_label
        self.class_names = df_label.columns



data_factory = {
                'Trackdata':Trackdata
                }
