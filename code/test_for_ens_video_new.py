
import logging
import pandas as pd
import json
import os
import pickle
import numpy as np
import csv

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

import torch
# from test_for_submit import test_save
# import test_for_submit
# import no_pipeline_eval_metrics
from no_pipeline_eval_metrics import merge_changepoints, test_save
# import no_pipeline_eval_metrics_msd
# from multimodel_test_for_submit import test_save, load
# import ensemble_task
from andi_datasets.utils_challenge import error_Ensemble_dataset, distribution_distance, models_phenom, multimode_dist
import pandas
import ruptures as rpt
from copy import deepcopy
from ensemble_task import load_from_trajectory_results, gmm_fit, get_weight

def read_data(exp_path,csv_path):
    with open(exp_path+csv_path, 'r') as fp:  #
        data = list(csv.reader(fp))  #
        data_set1 = []
        for i in range(len(data)):
            t = []
            del data[i][-1]    #删除最后一个空格字符串
            for j in range(len(data[i])):
                t.append(float(data[i][j]))
            data_set1.append(t)

    train_set=[]
    vip_index=[]

    for i in range(len(data_set1)):
        tmp=data_set1[i]
        vip_index.append(int(tmp[-1]))
        del tmp[-1]
        track_len=int(len(tmp)/2)

        desired_length = 20
        if track_len < desired_length:
            tmpx = tmp[:track_len]
            lastx= [tmpx[-1]]
            tmpx += lastx * (desired_length - track_len)
            tmpy = tmp[track_len:]
            lasty = [tmpy[-1]]
            tmpy += lasty * (desired_length - track_len)
            tmp = tmpx + tmpy
            track_len =int(len(tmp)/2)

        '''if track_len == 3:
            tmpx=tmp[:3]
            tmpx+=[tmpx[-1]]
            tmpy = tmp[3:]
            tmpy += [tmpy[-1]]
            tmp=tmpx+tmpy
            track_len+=1
        if track_len == 2:
            tmpx=tmp[:2]
            tmpx+=[tmpx[-1]]
            tmpx += [tmpx[0]]
            tmpy = tmp[2:]
            tmpy += [tmpy[-1]]
            tmpy += [tmpy[0]]
            tmp=tmpx+tmpy
            track_len+=2'''

        tmp += [0] * track_len
        tmp += [10] * track_len
        tmp += [0] * track_len
        train_set.append(tmp)

    return train_set,vip_index

def distribution_distance1(p:np.array, # distribution 1
                          q:np.array, # distribution 2
                          x:np.array = None, # support of the distributions (not needed for MAE)
                          metric = 'wasserstein', # distance metric (either 'wasserstein' or 'mae')
                          log = False, # log operation for calculating error of distribution D
                         )-> float:  # distance between distributions
    ''' Calculates distance between two distributions. '''
#     return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    if metric == 'mae':
        return np.abs(p-q).mean()
    elif metric == 'wasserstein':
        return wasserstein_distance(x, x, p, q)

def error_Ensemble_dataset1(true_data, pred_data,
                           size_support = int(1e6),
                           metric = 'wasserstein',
                           return_distributions = False,
                           only_dist = False,
                           K_range = [models_phenom().bound_D[0], models_phenom().bound_D[1]]):
    x_alpha = np.linspace(models_phenom().bound_alpha[0], 
                          models_phenom().bound_alpha[1], size_support)
    x_D = np.logspace(np.log10(K_range[0]), 
                      np.log10(K_range[1]), size_support)  
    
    
    dists = []
    for data in [true_data, pred_data]:
        
        if len(data.shape) > 1: # If we have more than one state
            alpha_info = np.delete(data, [2,3, -1], 0)
            d_info = data[2:-1,:]
            weights = data[-1,:]
            if weights.sum() > 1: weights /= weights.sum()
        else: # If single state
            alpha_info = data[:2]
            d_info = data[2:-1]
            weights = 1
            
        for idx, var in enumerate([alpha_info, d_info]):                                                
            dists.append(multimode_dist(var.T, weights, 
                                        bound  = models_phenom().bound_alpha if idx == 0 else models_phenom().bound_D, 
                                        x = x_alpha if idx == 0 else x_D))
    if only_dist:
        return dists
    # Distance between alpha dists
    distance_alpha = distribution_distance1(p = dists[0], q = dists[2],
                                           x = x_alpha, metric = metric)
    distance_D = distribution_distance1(p = dists[1], q = dists[3],
                                       x = x_D, metric = metric)
    
    if return_distributions:
        return distance_alpha, distance_D, dists
    else:
        return distance_alpha, distance_D
    


def distribution_distance_new(p:np.array, # distribution 1
                          q:np.array, # distribution 2
                          x:np.array = None, # support of the distributions (not needed for MAE)
                          metric = 'wasserstein' # distance metric (either 'wasserstein' or 'mae')
                         )-> float:  # distance between distributions
    ''' Calculates distance between two distributions. '''
#     return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    js_divergence = jensenshannon(p / p.sum() + 1e-12, q / q.sum() + 1e-12)
    return js_divergence
    # if metric == 'mae':
    #     return np.abs(p-q).mean()
    # elif metric == 'wasserstein':
    #     return wasserstein_distance(x, x, p, q)

def error_Ensemble_dataset_new(true_data, pred_data,
                           size_support = int(1e6),
                           metric = 'wasserstein',
                           return_distributions = False,
                           only_dist = False,
                           K_range = [models_phenom().bound_D[0], models_phenom().bound_D[1]],
                           log = False,
                           ):
    x_alpha = np.linspace(models_phenom().bound_alpha[0], 
                          models_phenom().bound_alpha[1], size_support)
    x_D = np.logspace(np.log10(K_range[0]), 
                      np.log10(K_range[1]), size_support)  
    
    
    dists = []
    for data in [true_data, pred_data]:
        
        if len(data.shape) > 1: # If we have more than one state
            alpha_info = np.delete(data, [2,3, -1], 0)
            d_info = data[2:-1,:]
            weights = data[-1,:]
            if weights.sum() > 1: weights /= weights.sum()
        else: # If single state
            alpha_info = data[:2]
            d_info = data[2:-1]
            weights = 1
            
        for idx, var in enumerate([alpha_info, d_info]):                                                
            dists.append(multimode_dist(var.T, weights, 
                                        bound  = models_phenom().bound_alpha if idx == 0 else models_phenom().bound_D, 
                                        x = x_alpha if idx == 0 else x_D, normalized=True))
    if only_dist:
        return dists
    # Distance between alpha dists
    distance_alpha = distribution_distance(p = dists[0], q = dists[2],
                                           x = x_alpha, metric = metric)
    distance_D = distribution_distance_new(p = dists[1], q = dists[3],
                                        x = x_D, metric = metric)
    
    if return_distributions:
        return distance_alpha, distance_D, dists
    else:
        return distance_alpha, distance_D







# addre_list = ['/data1/jiangy/andi_tcu/code/andi_2/output/_2024-11-12_15-20-50_4Yp/']
# notes_map = ['tamsd']

def test_for_ens(data_dir, addre, device, num_exp=22, num_fov=1):
    # data_dir = f'datasets/andi_set/data_ensemble/0121-3_multi_state/{date}-{data_idx}/'
    metrics = []
    addre_idx = 1
    data_idx = 1
    date = '0121'

    exp_skip = [0, 2, 3, 4, 5, 9, 12, 14, 16]
    # num_exp=22
    exp_range = range(num_exp)
    # num_fov=1
    fov_range = range(num_fov)
    preds_all = [[[] for j in fov_range] for i in exp_range]
    pads_all = [[[] for j in fov_range] for i in exp_range]

    if True:
        for i in exp_range:
            for j in fov_range:
                public_data_path = data_dir + 'video/exp_{}/'.format(i)
                csv_data_path = 'all_tracks_fov_{}.csv'.format(j)
                test_set,vip_index=read_data(exp_path=public_data_path,csv_path=csv_data_path)
                convert_csv_data_path = 'convert_all_trajs_fov_{}.csv'.format(j)
                convert_index_csv_data_path = 'convert_all_trajs_fov_index_{}.csv'.format(j)

                with open(os.path.join(public_data_path,convert_csv_data_path), 'w',
                        encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    ttt = test_set
                    writer.writerows(ttt)
                with open(os.path.join(public_data_path,convert_index_csv_data_path), 'w', encoding='utf-8',newline='') as f:
                    writer = csv.writer(f)
                    for item in vip_index:
                        writer.writerow([item])
                # public_data_path = '/data4/jiangy/AnDiChallenge/dataset/andi2_pilot_dataset/track_2/exp_{}/'.format(i)  # make sure the folder has this name or change it
                # csv_data_path = 'trajs_fov_{}.csv'.format(j)
                # convert_csv_data_path = 'convert_trajs_fov_{}.csv'.format(j)
                # test_set=read_data(exp_path=public_data_path,csv_path=csv_data_path)
                # with open(os.path.join(public_data_path,convert_csv_data_path), 'w',
                #           encoding='utf-8', newline='') as f:
                #     writer = csv.writer(f)
                #     ttt = test_set
                #     writer.writerows(ttt)
                # csv_path = '/data1/jiangy/andi_tcu/code/andi_2/datasets/andi_set/1025_multi_state/train/train0_Nall_1000_T200_200_K0_1.csv'
                csv_path = data_dir + 'video/exp_{}/convert_all_trajs_fov_{}.csv'.format(i, j)
                # csv_path = '/data4/jiangy/AnDiChallenge/dataset/andi2_pilot_dataset/track_2/exp_{}/convert_trajs_fov_{}.csv'.format(i, j)
                save_dir='./tmp/{}/track_1/exp_{}'.format(addre[-4:-1], i)
                # preds, pad = no_pipeline_eval_metrics.test_save(device, addre, csv_path, save_dir, exp=i, fov=j, addre_state=addre_state, addre_cp=addre_cp)
                preds, pad = test_save(device, addre, csv_path, save_dir, offset=-1, model_cp=False, return_preds=True)
                # print(len(preds), len(pad))
                preds_all[i][j] = preds
                pads_all[i][j] = pad
            
        save_predsall = np.asarray(preds_all, dtype=object)
        save_padsall = np.asarray(pads_all, dtype=object)
        os.makedirs('./tmp/{}/data_{}_model_{}/track1/'.format(date, data_idx, addre_idx), exist_ok=True)
        np.save('./tmp/{}/data_{}_model_{}/track1/ens_track_1_all_preds_all.npy'.format(date, data_idx, addre_idx), save_predsall)
        np.save('./tmp/{}/data_{}_model_{}/track1/ens_track_1_all_pad_all.npy'.format(date, data_idx, addre_idx), save_padsall)

        K_max_a = 5
        K_max_k = 4
        progressive_explore = False

        preds_all = np.load('./tmp/{}/data_{}_model_{}/track1/ens_track_1_all_preds_all.npy'.format(date, data_idx, addre_idx), allow_pickle=True)
        pads_all = np.load('./tmp/{}/data_{}_model_{}/track1/ens_track_1_all_pad_all.npy'.format(date, data_idx, addre_idx), allow_pickle=True)


        # 基于video的和基于traj的输出数值分布匹配

        def get_var(preds_all, pad_all):
            a_var = [[[] for j in fov_range] for i in exp_range]
            k_var = [[[] for j in fov_range] for i in exp_range]
            all_len = []
            all_log_k = []

            for i in exp_range:
                for j in fov_range:
                    pred = preds_all[i][j]
                    tmp_a = []
                    tmp_k = []
                    tmp_cp = []
                    tmp_cp_cnt = []
                    for idx in range(pred.shape[0]):
                        count_true = np.sum(pad_all[i][j][idx])
                        # print(count_true)
                        all_len.append(count_true)
                        all_log_k += pred[idx, :count_true, 1].tolist()
                        var_a = np.var(pred[idx, :count_true, 0])
                        var_k = np.var(pred[idx, :count_true, 1])
                        tmp_a.append(var_a)
                        tmp_k.append(var_k)
                    a_var[i][j] = tmp_a
                    k_var[i][j] = tmp_k
            return a_var, k_var, all_log_k

        # preds_all = np.load('../../challenge_results/daR/track_1_all_preds_all.npy', allow_pickle=True)
        # pad_all = np.load('../../challenge_results/daR/track_1_all_pad_all.npy', allow_pickle=True)
        a_var, k_var, all_log_k = get_var(preds_all, pads_all)

        # 待匹配的traj数据分布

        # preds_all_t2_all = np.load('../../challenge_results/daR/track_2_all_preds_all.npy', allow_pickle=True)
        # pad_all_t2_all = np.load('../../challenge_results/daR/track_2_all_pad_all.npy', allow_pickle=True)
        # a_var_t2_all, k_var_t2_all, all_log_k_t2_all = get_var(preds_all_t2_all, pad_all_t2_all)

        def histogram_matching(source, template):
            oldshape = source.shape
            source = source.ravel()
            template = template.ravel()
            
            s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
            t_values, t_counts = np.unique(template, return_counts=True)
            
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]
            
            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
            
            return interp_t_values[bin_idx].reshape(oldshape)

        # source = np.array(all_log_k) #np.random.normal(loc=0, scale=1, size=(N, T, 2))
        # template = np.array(all_log_k_t2_all) #np.random.normal(loc=5, scale=2, size=(N, T, 2))

        # all_log_k_matched = histogram_matching(source, template)


        preds_all = deepcopy(preds_all)
        pads_all = deepcopy(pads_all)
        # all_log_k = deepcopy(all_logk_t1_vip)
        # all_log_k = deepcopy(all_log_k_matched)
        # tmp = []
        # cnt = 0
        # for i in exp_range:
        #     for j in fov_range:
        #         if i == 0:
        #             continue
        #         pred = preds_all[i][j]
        #         for idx in range(pred.shape[0]):
        #             count_true = np.sum(pad_all[i][j][idx])
        #             print(count_true)
        #             pred[idx, :count_true, 1] = np.array(all_log_k[cnt:cnt+count_true]).reshape(-1, )
        #             cnt += count_true
        #             tmp += pred[idx, :count_true, 1].tolist()

        # a_var_t1_vip, k_var_t1_vip, all_log_k_t1_vip = get_var(preds_all_t1_vip, pad_all_t1_vip)

        # a_var = deepcopy(a_var_t2_all)
        # k_var = deepcopy(k_var_t2_all)
        # a_var, k_var, _ = get_var(preds_all, pad_all)
        percentile = 0.8 * 100
        # pen_map = {-7: 20, -6: 10, -5: 1, -4: 0.6, -3: 0.3, -2: 0.1, -1: 0.05}
        pen_map = {-11:80, -10: 80, -9: 60, -8: 30, -7: 20, -6: 10, -5: 5, -4: 2, -3: 1, -2: 0.5, -1: 0.2, 0: 0.1, 1: 0.05}
        ideal_cp = [1, 0.5, 0.1, 1, 1, 20, 20, 1, 20, 1, 1, 1, 5]
        # print('exp\tidea\tpercen_a\t\tpen_a\tpercen_k\t\tpen_k')
        min_percen = 10
        offset = -1
        for exp in exp_range:
            percen_a = np.percentile(np.array(sum(a_var[exp], [])), percentile)
            percen_k = np.percentile(np.array(sum(k_var[exp], [])), percentile)
            # print(exp, percen_a, percen_k)
            # min_percen = min(min_percen, round(np.log(percen_a)), round(np.log(percen_k)))
            # print(exp, ideal_cp[exp], percen_a, round(np.log(percen_a)), pen_map[round(np.log(percen_a))+offset], percen_k, round(np.log(percen_k)), pen_map[round(np.log(percen_k))+offset], sep='\t')
        # print(min_percen)
            
        # import csv

        pen_list = [(0,0) for _ in exp_range]
        offset
        for exp in exp_range:
            percen_a = np.percentile(np.array(sum(a_var[exp], [])), percentile)
            percen_k = np.percentile(np.array(sum(k_var[exp], [])), percentile)
            pen_list[exp] = (pen_map[round(np.log(percen_a))+offset], pen_map[round(np.log(percen_k))+offset])
            
            
        # print(pen_list)
        

        # pen_list[1] = (0.01, 0.01)
        # pen_list[2] = (5, 2)
        # pen_list[6] = (0.5, 0.5)
        # pen_list[7] = (0.5, 0.5)
        # pen_list[9] = (0.2, 0.2)

        cnt = [0, 0, 0, 0]
        cp_num = []

        filep_index = None
        # print(pen_list)
        for exp in exp_range:
            pen_a = pen_list[exp][0] #* 2
            pen_k = pen_list[exp][1] #* 4
            cp_tmp = []
            for fov in fov_range:
                results = []
                # print("Exp {} Fov {}".format(exp, fov))
                pred = preds_all[exp][fov]
                # print(pred.shape, pad_all[exp][fov].shape)
                for idx in range(pred.shape[0]):
                    # print(idx, end='\t')
                    count_true = np.sum(pads_all[exp][fov][idx])
                    pre_a = pred[idx, :count_true, 0].tolist()
                    pre_k = pred[idx, :count_true, 1].tolist()
                    model_a = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_a))
                    # breakpoints = model_a.predict(pen=30)
                    breakpoints_a = model_a.predict(pen=pen_a)
                    model_k = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_k))
                    breakpoints_k = model_k.predict(pen=pen_k)
                    # breakpoints = sorted(list(set(breakpoints_a).union(set(breakpoints_k))))
                    breakpoints = merge_changepoints(breakpoints_a, breakpoints_k)
                    if pen_k > 50:
                        breakpoints = breakpoints_a
                    # print(breakpoints)
                    cp_tmp.append(len(breakpoints)-1)

                    segments_a = [pre_a[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
                    segments_k = [pre_k[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]

                    tmp = []
                    tmp.append(idx)

                    for j in range(len(breakpoints)):
                        aver_k=sum(segments_k[j]) / len(segments_k[j])
                        aver_k=10 ** aver_k
                        # aver_k=np.exp(aver_k).item()
                        # tmp.append(aver_k)
                        aver_a = sum(segments_a[j]) / len(segments_a[j])
                        # tmp.append(aver_a)
                        aver_state=2
                        # if exp == 11:
                        #     if aver_a > 1:
                        #         aver_state = 3
                        #         aver_a = 1.9
                        if aver_a < 2e-3: # 5e-3
                            # print('alpha={} -> 0, state={} -> 0'.format(aver_a, aver_state))
                            aver_a = 0
                            aver_state = 0
                        elif aver_a < 0.02: # 0.05
                            # print('alpha={} < 0.05, state={} -> 1'.format(aver_a, aver_state))
                            # aver_a = 0
                            aver_state = 1
                        if aver_a > 1.88:
                            # print('alpha={} > 1.88, state={} -> 3'.format(aver_a, aver_state))
                            aver_state = 3
                            if aver_a > 1.99:
                                aver_a = 1.99
                        if aver_k < 2e-2:
                            # print('K={} -> 0, state={} -> 0, alpha={}'.format(aver_k, aver_state, aver_a))
                            aver_k = 0
                            aver_state = 0
                        cnt[aver_state] += 1
                        tmp.append(aver_k)
                        tmp.append(aver_a)
                        tmp.append(aver_state)
                        tmp.append(breakpoints[j])

                    results.append(tmp)
                
                save_dir = './tmp/{}/data_{}_model_{}/track1/exp_{}'.format(date, data_idx, addre_idx, exp)
                # save_dir = '../../results/daR_new/track_2_test_0707/exp_{}'.format(exp)
                os.makedirs(save_dir, exist_ok=True)

                file = open(save_dir + '/fov_{}.txt'.format(fov), 'w')

                # 定义一个数组

                # 将数组逐行写入txt文件，用逗号隔开
                for item1 in results:
                    for i,item2 in enumerate(item1):
                        if i != len(item1)-1:
                            file.write(str(item2) + ',')
                        else:
                            file.write(str(item2))
                    file.write('\n')
                # 关闭文件
                file.close()
            cp_num.append(cp_tmp)

        root = './tmp/{}/data_{}_model_{}/track1'.format(date, data_idx, addre_idx)
        results = load_from_trajectory_results(root, exp_range, fov_range)

        exp_range = range(num_exp)
        for exp in exp_range:
            K_max_a, K_max_k = 3, 3
            K_given = 3
            # K_max_a, K_max_k = K_map[exp]
            # print('Processing: ', exp)
            fov_res = results[exp]
            alpha = np.array(fov_res['alpha']).reshape(-1, )
            K = np.array(fov_res['K']).reshape(-1, )
            weights = None #np.array(fov_res['weight']).reshape(-1, )
            assigned_mu_alpha, assigned_sigma_alpha = gmm_fit(alpha, K_max=K_max_a, K_given=K_given, time_weight=weights)
            assigned_mu_K, assigned_sigma_K = gmm_fit(K, K_max=K_max_k, K_given=K_given, time_weight=weights)
            # joint = np.concatenate((alpha, K), axis=-1)
            # assigned_mu, assigned_sigma = gmm_fit(joint, K_max=5)
            cat_mu = np.concatenate((assigned_mu_alpha.reshape(-1,1), assigned_mu_K.reshape(-1,1)), axis=-1)
            cat_sigma = np.concatenate((assigned_sigma_alpha.reshape(-1,1), assigned_sigma_K.reshape(-1,1)), axis=-1)
            ensemble_results = get_weight(cat_mu, cat_sigma, fov_res, time_weighted=False if weights is None else True)
            # print('num: ', len(ensemble_results))
            file_path = os.path.join(root, 'exp_{}'.format(exp), 'ensemble_labels.txt')
            os.makedirs(os.path.join(root, 'exp_{}'.format(exp)), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
                writer = csv.writer(f, delimiter=';')
                # writer.writerows(ensemble_results)
                writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose]

            target_path = data_dir + 'video/exp_{}'.format(exp)
            true = np.loadtxt(os.path.join(target_path, 'ens_labs_fov_0.txt'), skiprows = 1, delimiter = ';')
            pred = np.loadtxt(os.path.join(root, 'exp_{}'.format(exp), 'ensemble_labels.txt'), skiprows = 1, delimiter = ';')

            # TODO: 修改D的数值范围到1e-3到1e3
            # distance_a_exp, distance_d_exp, dists = error_Ensemble_dataset1(true, pred, return_distributions = True)
            distance_a_exp, distance_d_exp, dists = error_Ensemble_dataset_new(true, pred, return_distributions = True, K_range=[1e-6, 1e3], size_support=int(1e5))

            avg_alpha, avg_d = [], []
            avg_alpha.append(distance_a_exp)
            avg_d.append(distance_d_exp)
            data_metrics = pandas.DataFrame(data = np.vstack((np.arange(len(avg_alpha)),avg_alpha, avg_d)).transpose(),
                                            columns = ['Exp', 'alpha', 'K'])
            data_metrics['Exp'] = data_metrics['Exp'].values.astype(int)
                
            # with open('./tmp/exp_{}/ens_metrics_1025.txt'.format(exp), 'w') as f:
            metrics.append({"Exp": exp, "alpha": avg_alpha[0], "K": avg_d[0], "model": addre, "data": data_dir})
            # with open(data_dir + 'ens_res/model_{}_exp_{}_ens_metrics.txt'.format(notes_map[addre_idx], exp), 'w') as f:
            #     f.write(str(np.mean(avg_alpha)) + '\n' + str(np.mean(avg_d)) + '\n' + str(data_metrics))


    df = pd.DataFrame(metrics)
    df.to_excel(data_dir + 'ens_metrics_{}_eval_test_0121_paper.xlsx'.format(data_idx), index=False)
    return metrics
    # with open(data_dir + 'ens_metrics_{}_eval_K-2_2.json'.format(data_idx), 'w') as f:
        # json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    logger = logging.getLogger('__main__')

    NEG_METRICS = {'loss'}  # metrics for which "better" is lesss

    val_times = {"total_time": 0, "count": 0}
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    data_dir = '/data1/jiangy/U-LFormer/data/0121-3_multi_state/0121-1/'
    addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-12-06_14-03-17_k9l/'
    test_for_ens(data_dir, addre, device)