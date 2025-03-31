import csv
# auxiliaries
import numpy as np
import matplotlib.pyplot as plt
from andi_datasets.utils_trajectories import plot_trajs, motion_blur, normalize, segs_inside_fov
from andi_datasets.utils_videos import transform_to_video, play_video, psf_width
from andi_datasets.models_phenom import models_phenom
from andi_datasets.utils_challenge import label_continuous_to_list, extract_ensemble, label_filter, df_to_array, get_VIP, file_nonOverlap_reOrg
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2, _defaults_andi2
from andi_datasets.datasets_phenom import datasets_phenom
from andi_datasets.datasets_theory import datasets_theory
import deeptrack as dt
import imageio
import random
import os
import argparse
from collections import Counter

from tqdm.auto import tqdm
import pandas as pd
import csv
import shutil
from pathlib import Path



def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K_low', default=-12, type=int)
    parser.add_argument('--K_high', default=6, type=int)
    parser.add_argument('--T_low', default=20, type=int)
    parser.add_argument('--T_high', default=50, type=int)
    parser.add_argument('--N_all', default=5e5, type=int)
    parser.add_argument('--N', default=1000, type=int)
    # parser.add_argument('--K_high', default=6, type=int)
    parser.add_argument('--model', default='multi_state', type=str)
    parser.add_argument('--generate_video', default=False, action='store_true')
    parser.add_argument('--motion_blur', default=False, action='store_true')
    parser.add_argument('--no_sample', default=False, action='store_true')
    parser.add_argument('--bg_mean', default=100, type=int)
    parser.add_argument('--particle_mean', default=500, type=int)
    parser.add_argument('--root_dir', default='./datasets/andi_set/1118_', type=str)
    parser.add_argument('--dist', default='undefined_3', type=str)
    args = parser.parse_args()
    return args


def generate_trajectories (N_all = 5000, T_min=50, T_max=200, L=1.8*128, D_max=1000, num_a_permodel=30, num_k_permodel=10000, N=100, log_range=None, args=None):
    '''
    # number of time steps per trajectory (frames)
    T_max = 200
    # number of trajectories       #长度在T_max和T_min内取值
    T_min = 200                #  全200的话就把这个也设置成200
    # Length of box (pixels)
    L = 1.8*128
    # diffusion coefficient (pixels^2 / frame)
    D_max = 1000     #diffusion coefficient从0-1000内随机取值，等间隔采样num_k_permodel   #这个可以仔细考虑一下  比赛里给的范围是0-10^6，均匀采样感觉范围太大了，一般D都不会超过100.目前的策略是0-1000均匀采样
    num_a_permodel=30   #diffusion coefficient从0-2内随机取值，等间隔采样num_a_permodel
    num_k_permodel=10000
    N_all = 5000
    N=100   #每确定一组参数（T，K，a）生成100条轨迹
    #
    '''
    K_log_range = [-12, 6] if log_range is None else log_range
    # exponents = [2 * i / num_a_permodel for i in range(0, num_a_permodel + 1)]
    # K_diffusion = [100 * i / num_k_permodel for i in range(0, num_k_permodel + 1)]
    exponents = [0.001 + 1.89 * i / num_a_permodel for i in range(0, num_a_permodel + 1)] + [1.89 + 0.1 * i / 5 for i in range(0, 5 + 1)]
    if args.model == 'directed':
        num_a_permodel = 10
        exponents = [1.89 + 0.1 * i / num_a_permodel for i in range(0, num_a_permodel + 1)]
    K_diffusion = [pow(10, K_log_range[0]) * pow(10, (K_log_range[1] - K_log_range[0]) * i / num_k_permodel) for i in range(0, num_k_permodel)] # 1e-12 -- 1e6 之间取值
    T=[T_min+i for i in range(0, T_max-T_min + 1)]
    radius = [i for i in range(1, int(L // 16))]

    dics = []

    train_set= []  #  #用于存放轨迹的list    len(trajs)=N_all     第i行：   x1，x2，...,x_T; y1,y2,...,y_T; a1，a2，...,a_T; k1,k2,...,k_T    T=len(trajs[i])/4
    for i in range (int(N_all/N)):
        dic = _get_dic_andi2(2)
        ensemble_alpha = []
        ensemble_K = []

        random_number_T = random.randint(0, T_max-T_min)
        T_1=T[random_number_T]
        if args.generate_video and args.motion_blur:
            T_1 = T_1 * oversamp_factor

        random_number_a = random.randint(0, num_a_permodel)
        a_1 = exponents[random_number_a]
        random_number_a = random.randint(0, num_a_permodel)
        a_2 = exponents[random_number_a]
        random_number_a = random.randint(0, num_a_permodel)
        a_3 = exponents[random_number_a]

        random_number_k = random.randint(0, num_k_permodel - 1)
        k_1 = K_diffusion[random_number_k]
        random_number_k = random.randint(0, num_k_permodel - 1)
        k_2 = K_diffusion[random_number_k]
        random_number_k = random.randint(0, num_k_permodel - 1)
        k_3 = K_diffusion[random_number_k]

        if args.model == 'immobile_traps':
            r_1 = random.sample(radius, 1)[0]
            Nt = min(int(L / r_1), 10)
            trajs_model2, labels_model2 = models_phenom().immobile_traps(N = N,
                                                                        T = T_1,                
                                                                        L = L,
                                                                        r = r_1,
                                                                        Pu = 0.1, 
                                                                        Pb = 0.01, 
                                                                        Ds = [k_1, 0], 
                                                                        alphas = [a_1, 0], 
                                                                        Nt = Nt,
                                                                        # traps_pos: np.array = None,
                                                                        # deltaT = 1
                                                                        )
            ensemble_alpha += [a_1]
            ensemble_K += [k_1]

        elif args.model == 'multi_state' or args.model == 'directed':
            if args.dist.startswith('near'):
                k_2 = k_1 * 0.5
                k_3 = k_1 * 1.5
            elif args.dist.startswith('far'):
                k_1 = random.uniform(0.5, 2)
                k_2 = k_1 / 10
                k_3 = k_1 * 10
            trajs_model2, labels_model2 = models_phenom().multi_state(N=N,
                                                                    L=L,
                                                                    T=T_1,
                                                                    epsilon_a=[0, 0],
                                                                    gamma_d=[1, 1],
                                                                    alphas=[[a_1, 0], [a_2, 0], [a_3, 0]],
                                                                    # Fixed alpha for each state
                                                                    Ds=[[k_1, 0], [k_2, 0], [k_3, 0]],
                                                                    # Mean and variance of each state
                                                                    M=[[0.99, 0.005, 0.005], [0.005, 0.99, 0.005],
                                                                        [0.005, 0.005, 0.99]]
                                                                    )
            ensemble_alpha += [a_1, a_2, a_3]
            ensemble_K += [k_1, k_2, k_3]
            if args.dist[-1] == '3':
                dic.update({'N': N, 'L': L, 'T': T_1, 'epsilon_a': [0, 0],
                                                    'gamma_d': [1, 1],
                                                    'alphas': np.array([[a_1, 0.1*a_1], [a_2, 0.1*a_2], [a_3, 0.1*a_3]]),
                                                    # Fixed alpha for each state
                                                    'Ds': np.array([[k_1, 0.1*k_1], [k_2, 0.1*k_2], [k_3, 0.1*k_3]]),
                                                    # Mean and variance of each state
                                                    'M': np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005],
                                                        [0.005, 0.005, 0.99]]),
                                                        })
            elif args.dist[-1] == '2':
                dic.update({'N': N, 'L': L, 'T': T_1, 'epsilon_a': [0, 0],
                                                    'gamma_d': [1, 1],
                                                    'alphas': np.array([[a_1, 0.1*a_1], [a_2, 0.1*a_2]]),
                                                    # Fixed alpha for each state
                                                    'Ds': np.array([[k_1, 0.1*k_1], [k_2, 0.1*k_2]]),
                                                    # Mean and variance of each state
                                                    'M': np.array([[0.99, 0.01], [0.01, 0.99],]),
                                                        })
            dics.append(dic)

        elif args.model == 'dimerization':
            r_1 = random.sample(radius, 1)[0]
            Nt = min(int(L / r_1), 10)
            trajs_model2, labels_model2 = models_phenom().dimerization(N = N,
                                                                        T = T_1,                
                                                                        L = L,
                                                                        r = r_1,
                                                                        Pu = 0.1, 
                                                                        Pb = 0.01,
                                                                        Ds = [[k_1, 0], [k_2, 0]],
                                                                        alphas = [[a_1, 0], [a_2, 0]],
                                                                        )
            ensemble_alpha += [a_1, a_2]
            ensemble_K += [k_1, k_2]

        elif args.model == 'confinement':
            r_1 = random.sample(radius, 1)[0]
            number_compartments = 50
            radius_compartments = 10
            compartments_center = models_phenom._distribute_circular_compartments(Nc = number_compartments, 
                                                                                r = radius_compartments,
                                                                                L = L # size of the environment
                                                                                ) 
            trajs_model2, labels_model2 = models_phenom().confinement(N = N,
                                                                        T = T_1,
                                                                        L = L,
                                                                        Ds = [[k_1, 0], [k_2, 0]], 
                                                                        alphas = [[a_1, 0], [a_2, 0]],
                                                                        gamma_d = [1], 
                                                                        epsilon_a = [0],
                                                                        r = radius_compartments,
                                                                        comp_center = compartments_center,
                                                                        Nc = 10,
                                                                        trans = 0.1, 
                                                                        )
            ensemble_alpha += [a_1, a_2]
            ensemble_K += [k_1, k_2]

        # def get_weight(alpha, K):
        #     # weight = np.zeros((np.unique(assigned_mu).shape[0], 1))
        #     print(alpha.shape)
        #     weight_map = {}
        #     time_sum = 0
        #     for i in range(alpha.shape[0]):
        #         key = '{}_{}_{}_{}'.format(alpha[i], 0, K[i], 0)
        #         if key not in weight_map:
        #             weight_map[key] = 1
        #         else:
        #             weight_map[key] += 1
        #         time_sum += 1
        #     print(weight_map)
        #     results = []
        #     for key, value in weight_map.items():
        #         keys = key.split('_')
        #         # print(keys)
        #         results.append([float(keys[0]), float(keys[1]), float(keys[2]), float(keys[3]), value / time_sum])
        #     return results
        # ensemble_results = get_weight(np.array(ensemble_alpha), np.array(ensemble_K))
            
        # with open(os.path.join(root, 'ens_gt', f'ensemble_labels_{i}.txt'), 'w') as f:
        #     f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
        #     writer = csv.writer(f, delimiter=';')
        #     # writer.writerows(ensemble_results)
        #     writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose]

        # # os.path.join(root, 'video', f'{i}.tiff')
        # generate_video_from_trajs(trajs_model2, save_path=os.path.join(root, 'video', f'{i}.tiff'), vip_num=random.sample(range(1, N), 10), output_length=T_1 / oversamp_factor, oversamp_factor=oversamp_factor, exposure_time=exposure_time)
        
        #要把所有的轨迹concatenate起来
        # 使用transpose交换第一个和第二个维度
        trajs_tmp = trajs_model2.transpose(1, 0, 2)
        labels_tmp = labels_model2.transpose(1, 0, 2)
        trajs_tmp = trajs_tmp.tolist()
        labels_tmp = labels_tmp.tolist()

        for i in range (len(trajs_tmp)):
            traj_x = [item[0] for item in trajs_tmp[i]]
            traj_y = [item[1] for item in trajs_tmp[i]]
            label_a= [item[0] for item in labels_tmp[i]]
            label_k = [item[1] for item in labels_tmp[i]]
            label_state = [item[2] for item in labels_tmp[i]]

            traj_i = traj_x + traj_y +label_a + label_k + label_state


            train_set.append(traj_i)
    
    print(len(dics))
    PATH = os.path.join(root, 'video')

    generate_videos = True # if you don't want to produce videos, set this to False

    # challenge_phenom_dataset(save_data = True,
    # get_challenge_dataset_new(save_data = True,
    #                         dics = dics,
    #                         path = PATH, 
    #                         return_timestep_labs = True,
    #                         num_fovs = 1, # because of non-overlapping fovs, this should be 1
    #                         num_vip=1,
    #                         get_video = generate_videos,
    #                         particle_prop=[args.particle_mean,20],
    #                         bg_prop=[args.bg_mean, 0],
    #                         );

    return train_set

# 修改了challenge_phenom_dataset，来自andi_datasets.datasets_challenge，方便修改部分参数
# 默认的"particle_intensity": [500,20,]
# 默认的"background_mean": 100, "background_std": 0,  
def get_challenge_dataset(experiments = 5,
                             dics = None,
                             repeat_exp = True,
                             num_fovs = 1,
                             return_timestep_labs = False,
                             save_data = False,
                             path = 'data/',
                             prefix = '',
                             particle_prop = [500, 20],
                             bg_prop = [100, 0],
                             get_video = False, num_vip = None, get_video_masks = False,
                             files_reorg = False, path_reorg = 'ref/', save_labels_reorg = False, delete_raw = False):
    # Set prefixes for saved files
    if save_data:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        pf_labs_traj = prefix+'traj_labs'
        pf_labs_ens = prefix+'ens_labs'
        pf_trajs = prefix+'trajs'
        pf_videos = prefix+'videos'

    # Sets the models of the experiments that will be output by the function
    if dics is None:
        if isinstance(experiments, int):
            if repeat_exp: # If experiments can be repeated, we just sample randomly
                model_exp = np.random.randint(len(datasets_phenom().avail_models_name), size = experiments)
            else: # If not, we sampled them in an ordered way
                if experiments >= len(datasets_phenom().avail_models_name):
                    num_repeats = (experiments % len(datasets_phenom().avail_models_name))+1
                else:
                    num_repeats = 1
                model_exp = np.tile(np.arange(len(datasets_phenom().avail_models_name)), num_repeats)[:experiments]
            # We add one to get into non-Python numeration
            model_exp += 1
        else:
            model_exp = experiments
    # If list of dics is given, then just create a list of length = len(dics)
    else: 
        model_exp = [0]*len(dics)

    # Output lists
    trajs_out, labels_traj_out, labels_ens_out = [], [], []
    for idx_experiment, model in enumerate(tqdm(model_exp)):       

        ''' Generate the trajectories '''
        if dics is None:
            dic = _get_dic_andi2(model)
        else:
            dic = dics[idx_experiment]
            # Overide the info about model
            model = datasets_phenom().avail_models_name.index(dic['model'])+1    
        print(f'Creating dataset for Exp_{idx_experiment} ('+dic['model']+').')
        trajs, labels = datasets_phenom().create_dataset(dics = dic)            
        
        
        ''' Apply the FOV '''
        for fov in range(num_fovs):
            # Checking if file exist and creating an error
            if save_data:
                path_labs_traj = path/(pf_labs_traj+f'_exp_{idx_experiment}_fov_{fov}.txt')
                path_labs_ens = path/(pf_labs_ens+f'_exp_{idx_experiment}_fov_{fov}.txt')
                if path_labs_traj.exists() or path_labs_ens.exists():
                    raise FileExistsError(f'Target files for experiment {idx_experiment} and FOV {fov}. Delete the file or change path/prefix.')            


            # We take as min/max for the fovs a 5 % distance of L
            dist = 0.05
            min_fov = int(dist*_defaults_andi2().L)
            max_fov = int((1-dist)*_defaults_andi2().L)-_defaults_andi2().FOV_L
            # sample the position of the FOV
            fov_origin = (np.random.randint(min_fov, max_fov), np.random.randint(min_fov, max_fov))

            ''' Go over trajectories in FOV (copied from utils_trajectories for efficiency) '''
            trajs_fov, array_labels_fov, list_labels_fov, idx_segs_fov, frames_fov = [], [], [], [], []
            idx_seg = -1

            # Total frames
            frames = np.arange(trajs.shape[0])
            # We save the correspondance between idx in FOV and idx in trajs dataset
            for traj, label in zip(trajs[:, :, :].transpose(1,0,2),
                                   labels[:, :, :].transpose(1,0,2)):
                                
                nan_segms = segs_inside_fov(traj[:,:2], # take only the 2D projection of the traj
                                            fov_origin = fov_origin,
                                            fov_length = _defaults_andi2().FOV_L,
                                            cutoff_length = _defaults_andi2()._min_T)

                if nan_segms is not None:
                    for idx_nan in nan_segms:  
                        idx_seg+= 1  
                        
                        trajs_fov.append(traj[idx_nan[0]:idx_nan[1]])
                        frames_fov.append(frames[idx_nan[0]:idx_nan[1]])

                        lab_seg = []
                        for idx_lab in range(labels.shape[-1]):
                            lab_seg.append(_defaults_andi2().label_filter(label[idx_nan[0]:idx_nan[1], idx_lab]))
                        lab_seg = np.vstack(lab_seg).transpose()                    
                        array_labels_fov.append(lab_seg)

                        # Tranform continuous labels to list for correct output
                        if model == 2 or model == 4: 
                            # if multi-state or dimerization, we get rid of the label of state numbering
                            CP, alphas, Ds, states = label_continuous_to_list(lab_seg[:, :-1])
                        else:
                            CP, alphas, Ds, states = label_continuous_to_list(lab_seg)
                        
                        # Extract final point of trajectory 
                        T = CP[-1]
                        CP = CP[:-1]
                        list_gt = [idx_seg, Ds[0], alphas[0], states[0]]
                        for gtc, gta, gtd, gts in zip(CP, alphas[1:], Ds[1:], states[1:]):
                            list_gt += [gtc, gtd, gta, gts]
                        # Add end point of trajectory
                        list_gt.append(T)
                        list_labels_fov.append(list_gt)     

                        if save_data:
                            with open(path_labs_traj, 'a') as f:
                                writer = csv.writer(f, delimiter=',', lineterminator='\n',)
                                writer.writerow(list_gt)

                        # Save index of segment with its length to latter append in the dataframe              
                        idx_segs_fov.append(np.ones(trajs_fov[-1].shape[0])*idx_seg)             
            
            '''Extract ensemble trajectories''' 
            if len(array_labels_fov):
                ensemble_fov = extract_ensemble(np.concatenate(array_labels_fov)[:, -1], dic)
            else:
                raise Exception("No sufficiently long trajectory could be found inside the FOV. "
                                "This can be due to having too few particles or these moving too "
                                "fast and exiting the FOV before the minimal trajectory length is "
                                "reached, for instance. Try adjusting the experimental parameters "
                                "to match the FOV's scale. This can also be the result of a (very) "
                                "unlucky run, which can be alleviated with more and longer trajectories.")

            df_data = np.hstack((np.expand_dims(np.concatenate(idx_segs_fov), axis=1),
                                 np.expand_dims(np.concatenate(frames_fov), axis=1).astype(int),
                                 np.concatenate(trajs_fov)))            
            
            if 'dim' in dic.keys() and dic['dim'] == 3:
                df_traj = pd.DataFrame(df_data, columns = ['traj_idx', 'frame', 'x', 'y','z'])
            else:                
                df_traj = pd.DataFrame(df_data, columns = ['traj_idx', 'frame', 'x', 'y'])
            
            
            if get_video:
                print(f'Generating video for EXP {idx_experiment} FOV {fov}')               
                
                pad = -20 #  padding has to be further enough from the FOV so that the PSF 
                          # of particles does not enter in the videos
                array_traj_fov = df_to_array(df_traj.copy(), pad = pad)
                min_distance = psf_width()
                idx_vip = get_VIP(array_traj_fov, num_vip = num_vip,
                              min_distance_part = min_distance, 
                              min_distance_bound = min_distance,
                              boundary_origin = fov_origin,
                              boundary = _defaults_andi2().FOV_L,                    
                              pad = pad)  
                np.savetxt(path/(prefix+f'vip_idx_exp_{idx_experiment}_fov_{fov}.txt'), idx_vip)
                
                if not save_data:
                    pf_videos = ''                                
                
                video_fov = transform_to_video(array_traj_fov, # see that we insert the trajectories without noise!
                                               optics_props={
                                                   "output_region":[fov_origin[0], fov_origin[1],
                                                                    fov_origin[0] + _defaults_andi2().FOV_L, fov_origin[1] + _defaults_andi2().FOV_L]
                                                },
                                               get_vip_particles=idx_vip,
                                               with_masks = get_video_masks,
                                               save_video = save_data,
                                               path = path/(pf_videos+f'_exp_{idx_experiment}_fov_{fov}.tiff'),
                                               particle_props={"particle_intensity": particle_prop},
                                               background_props={"background_mean": bg_prop[0], "background_std": bg_prop[1]}, 
                                              ) 
                
                try:
                    videos_out.append(video_fov)
                except:
                    videos_out = [video_fov] 
                                        
            # Add noise to the trajectories (see that this has to be done
            # after the videos, so these are not affected by the noise).
            df_traj.x += np.random.randn(df_traj.shape[0])*_defaults_andi2().sigma_noise 
            df_traj.y += np.random.randn(df_traj.shape[0])*_defaults_andi2().sigma_noise                         
            if 'dim' in dic.keys() and dic['dim'] == 3:
                df_traj.z += np.random.randn(df_traj.shape[0])*_defaults_andi2().sigma_noise 
                
            
                    
            if return_timestep_labs:
                array_labels_fov = np.concatenate(array_labels_fov)
                df_traj['alpha'] = array_labels_fov[:, 0]
                df_traj['D'] = array_labels_fov[:, 1]
                df_traj['state'] = array_labels_fov[:, 2]

            if save_data:
                # Trajectories                    
                df_traj.to_csv(path/(pf_trajs+f'_exp_{idx_experiment}_fov_{fov}.csv'), index = False)
                # Ensemble labels
                with open(path_labs_ens, 'a') as f:
                    if model == 2: num_states = dic['alphas'].shape[0]
                    elif model == 1: num_states = 1
                    else: num_states = 2
                    model_n = dic['model']
                    f.write(f'model: {model_n}; num_state: {num_states} \n')
                    np.savetxt(f, ensemble_fov, delimiter = ';')
                    
            
            # Add data to main lists (trajectories and lists with labels)   
            trajs_out.append(df_traj)
            labels_traj_out.append(list_labels_fov)
            labels_ens_out.append(ensemble_fov)
            
    # If asked, create a reorganized version of the folders
    if files_reorg:
        file_nonOverlap_reOrg(raw_folder = path,
                              target_folder = path/path_reorg,
                              experiments = np.arange(len(model_exp)), # this only needs to be array, content does not matter
                              num_fovs = num_fovs,
                              save_labels = save_labels_reorg,
                              tracks = [2] if not get_video else [1,2],
                              print_percentage = False)
        
        if delete_raw:
            for item_path in path.iterdir():
                if item_path != path_reorg:  # The -1 deletes the compulsory / of the path
                    if item_path.is_dir():
                        shutil.rmtree(item_path)  # Remove directories
                    else:
                        item_path.unlink()  # Remove files
    
    if get_video:
        return trajs_out, videos_out, labels_traj_out, labels_ens_out
    else:
        return trajs_out, labels_traj_out, labels_ens_out

def get_challenge_dataset_new(experiments = 5,
                             dics = None,
                             repeat_exp = True,
                             num_fovs = 1,
                             return_timestep_labs = False,
                             save_data = False,
                             path = 'data/',
                             prefix = '',
                             particle_prop = [500, 20],
                             bg_prop = [100, 0],
                             get_video = False, num_vip = None, get_video_masks = False,
                             files_reorg = False, path_reorg = 'ref/', save_labels_reorg = False, delete_raw = False):
    # Set prefixes for saved files
    if save_data:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        pf_labs_traj = prefix+'traj_labs'
        pf_labs_ens = prefix+'ens_labs'
        pf_trajs = prefix+'trajs'
        pf_videos = prefix+'videos'

    # Sets the models of the experiments that will be output by the function
    if dics is None:
        if isinstance(experiments, int):
            if repeat_exp: # If experiments can be repeated, we just sample randomly
                model_exp = np.random.randint(len(datasets_phenom().avail_models_name), size = experiments)
            else: # If not, we sampled them in an ordered way
                if experiments >= len(datasets_phenom().avail_models_name):
                    num_repeats = (experiments % len(datasets_phenom().avail_models_name))+1
                else:
                    num_repeats = 1
                model_exp = np.tile(np.arange(len(datasets_phenom().avail_models_name)), num_repeats)[:experiments]
            # We add one to get into non-Python numeration
            model_exp += 1
        else:
            model_exp = experiments
    # If list of dics is given, then just create a list of length = len(dics)
    else: 
        model_exp = [0]*len(dics)

    # Output lists
    trajs_out, labels_traj_out, labels_ens_out = [], [], []
    for idx_experiment, model in enumerate(tqdm(model_exp)):       

        ''' Generate the trajectories '''
        if dics is None:
            dic = _get_dic_andi2(model)
        else:
            dic = dics[idx_experiment]
            # Overide the info about model
            model = datasets_phenom().avail_models_name.index(dic['model'])+1    
        print(f'Creating dataset for Exp_{idx_experiment} ('+dic['model']+').')
        trajs, labels = datasets_phenom().create_dataset(dics = dic)            
        
        
        ''' Apply the FOV '''
        for fov in range(num_fovs):
            # Checking if file exist and creating an error
            if save_data:
                path_labs_traj = path/(pf_labs_traj+f'_exp_{idx_experiment}_fov_{fov}.txt')
                path_labs_ens = path/(pf_labs_ens+f'_exp_{idx_experiment}_fov_{fov}.txt')
                if path_labs_traj.exists() or path_labs_ens.exists():
                    raise FileExistsError(f'Target files for experiment {idx_experiment} and FOV {fov}. Delete the file or change path/prefix.')            


            # We take as min/max for the fovs a 5 % distance of L
            dist = 0.05
            min_fov = int(dist*_defaults_andi2().L)
            max_fov = int((1-dist)*_defaults_andi2().L)-_defaults_andi2().FOV_L
            # sample the position of the FOV
            fov_origin = (np.random.randint(min_fov, max_fov), np.random.randint(min_fov, max_fov))

            ''' Go over trajectories in FOV (copied from utils_trajectories for efficiency) '''
            trajs_fov, array_labels_fov, list_labels_fov, idx_segs_fov, frames_fov = [], [], [], [], []
            idx_seg = -1

            # Total frames
            frames = np.arange(trajs.shape[0])
            # We save the correspondance between idx in FOV and idx in trajs dataset
            for traj, label in zip(trajs[:, :, :].transpose(1,0,2),
                                   labels[:, :, :].transpose(1,0,2)):
                                
                nan_segms = segs_inside_fov(traj[:,:2], # take only the 2D projection of the traj
                                            fov_origin = fov_origin,
                                            fov_length = _defaults_andi2().FOV_L,
                                            cutoff_length = _defaults_andi2()._min_T)

                if nan_segms is not None:
                    for idx_nan in nan_segms:  
                        idx_seg+= 1  
                        
                        trajs_fov.append(traj[idx_nan[0]:idx_nan[1]])
                        frames_fov.append(frames[idx_nan[0]:idx_nan[1]])

                        lab_seg = []
                        for idx_lab in range(labels.shape[-1]):
                            lab_seg.append(_defaults_andi2().label_filter(label[idx_nan[0]:idx_nan[1], idx_lab]))
                        lab_seg = np.vstack(lab_seg).transpose()                    
                        array_labels_fov.append(lab_seg)

                        # Tranform continuous labels to list for correct output
                        if model == 2 or model == 4: 
                            # if multi-state or dimerization, we get rid of the label of state numbering
                            CP, alphas, Ds, states = label_continuous_to_list(lab_seg[:, :-1])
                        else:
                            CP, alphas, Ds, states = label_continuous_to_list(lab_seg)
                        
                        # Extract final point of trajectory 
                        T = CP[-1]
                        CP = CP[:-1]
                        list_gt = [idx_seg, Ds[0], alphas[0], states[0]]
                        for gtc, gta, gtd, gts in zip(CP, alphas[1:], Ds[1:], states[1:]):
                            list_gt += [gtc, gtd, gta, gts]
                        # Add end point of trajectory
                        list_gt.append(T)
                        list_labels_fov.append(list_gt)     

                        if save_data:
                            with open(path_labs_traj, 'a') as f:
                                writer = csv.writer(f, delimiter=',', lineterminator='\n',)
                                writer.writerow(list_gt)

                        # Save index of segment with its length to latter append in the dataframe              
                        idx_segs_fov.append(np.ones(trajs_fov[-1].shape[0])*idx_seg)             
            
            '''Extract ensemble trajectories''' 
            if len(array_labels_fov):
                ensemble_fov = extract_ensemble(np.concatenate(array_labels_fov)[:, -1], dic)
            else:
                raise Exception("No sufficiently long trajectory could be found inside the FOV. "
                                "This can be due to having too few particles or these moving too "
                                "fast and exiting the FOV before the minimal trajectory length is "
                                "reached, for instance. Try adjusting the experimental parameters "
                                "to match the FOV's scale. This can also be the result of a (very) "
                                "unlucky run, which can be alleviated with more and longer trajectories.")

            df_data = np.hstack((np.expand_dims(np.concatenate(idx_segs_fov), axis=1),
                                 np.expand_dims(np.concatenate(frames_fov), axis=1).astype(int),
                                 np.concatenate(trajs_fov)))            
            
            if 'dim' in dic.keys() and dic['dim'] == 3:
                df_traj = pd.DataFrame(df_data, columns = ['traj_idx', 'frame', 'x', 'y','z'])
            else:                
                df_traj = pd.DataFrame(df_data, columns = ['traj_idx', 'frame', 'x', 'y'])
            
            
            if get_video:
                print(f'Generating video for EXP {idx_experiment} FOV {fov}')               
                
                pad = -20 #  padding has to be further enough from the FOV so that the PSF 
                          # of particles does not enter in the videos
                array_traj_fov = df_to_array(df_traj.copy(), pad = pad)
                min_distance = psf_width()
                idx_vip = get_VIP(array_traj_fov, num_vip = num_vip,
                              min_distance_part = min_distance, 
                              min_distance_bound = min_distance,
                              boundary_origin = fov_origin,
                              boundary = _defaults_andi2().FOV_L,                    
                              pad = pad)  
                np.savetxt(path/(prefix+f'vip_idx_exp_{idx_experiment}_fov_{fov}.txt'), idx_vip)
                
                if not save_data:
                    pf_videos = ''                                
                # print(trajs.shape, array_traj_fov.shape) # 200, 40, 2
                lens = array_traj_fov.shape[1]
                sample_idx = random.sample(range(lens), int(0.2 * lens)+1)
                # print(lens, sample_idx)
                array_traj_fov_sample = array_traj_fov[:, sample_idx, :]
                for snr, particle_prop in zip([2, 5], [[120, 20], [500, 20]]):
                    for density, array in zip(['low', 'high'], [array_traj_fov, array_traj_fov_sample]):
                        video_fov = transform_to_video(array,#array_traj_fov, # see that we insert the trajectories without noise!
                                               optics_props={
                                                   "output_region":[fov_origin[0], fov_origin[1],
                                                                    fov_origin[0] + _defaults_andi2().FOV_L, fov_origin[1] + _defaults_andi2().FOV_L]
                                                },
                                               get_vip_particles=idx_vip,
                                               with_masks = get_video_masks,
                                               save_video = save_data,
                                               path = path/(pf_videos+f'_exp_{idx_experiment}_fov_{fov}_snr_{snr}_dens_{density}.tiff'),
                                               particle_props={"particle_intensity": particle_prop},
                                               background_props={"background_mean": bg_prop[0], "background_std": bg_prop[1]}, 
                                              ) 
                
                try:
                    videos_out.append(video_fov)
                except:
                    videos_out = [video_fov] 
                                        
            # Add noise to the trajectories (see that this has to be done
            # after the videos, so these are not affected by the noise).
            df_traj.x += np.random.randn(df_traj.shape[0])*_defaults_andi2().sigma_noise 
            df_traj.y += np.random.randn(df_traj.shape[0])*_defaults_andi2().sigma_noise                         
            if 'dim' in dic.keys() and dic['dim'] == 3:
                df_traj.z += np.random.randn(df_traj.shape[0])*_defaults_andi2().sigma_noise 
                
            
                    
            if return_timestep_labs:
                array_labels_fov = np.concatenate(array_labels_fov)
                df_traj['alpha'] = array_labels_fov[:, 0]
                df_traj['D'] = array_labels_fov[:, 1]
                df_traj['state'] = array_labels_fov[:, 2]

            if save_data:
                # Trajectories                    
                df_traj.to_csv(path/(pf_trajs+f'_exp_{idx_experiment}_fov_{fov}.csv'), index = False)
                # Ensemble labels
                with open(path_labs_ens, 'a') as f:
                    if model == 2: num_states = dic['alphas'].shape[0]
                    elif model == 1: num_states = 1
                    else: num_states = 2
                    model_n = dic['model']
                    f.write(f'model: {model_n}; num_state: {num_states} \n')
                    np.savetxt(f, ensemble_fov, delimiter = ';')
                    
            
            # Add data to main lists (trajectories and lists with labels)   
            trajs_out.append(df_traj)
            labels_traj_out.append(list_labels_fov)
            labels_ens_out.append(ensemble_fov)
            
    # If asked, create a reorganized version of the folders
    if files_reorg:
        file_nonOverlap_reOrg(raw_folder = path,
                              target_folder = path/path_reorg,
                              experiments = np.arange(len(model_exp)), # this only needs to be array, content does not matter
                              num_fovs = num_fovs,
                              save_labels = save_labels_reorg,
                              tracks = [2] if not get_video else [1,2],
                              print_percentage = False)
        
        if delete_raw:
            for item_path in path.iterdir():
                if item_path != path_reorg:  # The -1 deletes the compulsory / of the path
                    if item_path.is_dir():
                        shutil.rmtree(item_path)  # Remove directories
                    else:
                        item_path.unlink()  # Remove files
    
    if get_video:
        return trajs_out, videos_out, labels_traj_out, labels_ens_out
    else:
        return trajs_out, labels_traj_out, labels_ens_out




def generate_video_from_trajs(trajs, save_path, vip_num, output_length=None, oversamp_factor=None, exposure_time=None):
    # MB = motion_blur(output_length = output_length, oversamp_factor = oversamp_factor, exposure_time = exposure_time)
    MB = None
    video = transform_to_video(trajs,
                               motion_blur_generator=MB,
                                save_video=True,
                                path=save_path,
                                with_masks=False,
                                get_vip_particles=vip_num,
                                particle_props = {"z": 0},
                                # background_props = {"background_mean": 0,
                                #                     "background_std": 0}, 
                                optics_props = {'output_region': [0,0,L,L]},
                                )
    # imageio.mimwrite(save_path, video)
    


args = get_args_parser()
print(args)

# number of time steps per trajectory (frames)
T_max = args.T_high
# number of trajectories       #长度在T_max和T_min内取值
T_min = args.T_low               #  全200的话就把这个也设置成200 
# 20-50 5w*1.1 50-100 20w*1.1 100-200 40w*1.1
# 运动模糊
oversamp_factor = 10
exposure_time = 0.2

# Length of box (pixels)
L = 1.8*128
# diffusion coefficient (pixels^2 / frame)
D_max = 1000     #diffusion coefficient从0-1000内随机取值，等间隔采样num_k_permodel   #这个可以仔细考虑一下  比赛里给的范围是0-10^6，均匀采样感觉范围太大了，一般D都不会超过100.目前的策略是0-1000均匀采样
num_a_permodel=30   #diffusion coefficient从0-2内随机取值，等间隔采样num_a_permodel
num_k_permodel=10000
N=args.N   #每确定一组参数（T，K，a）生成100条轨迹
# N=1000


N_all_train = args.N_all
# N_all_test = 5e5
per_num = 5e5
ratio = 0.1
idx = 0
# while idx < N_all_train:
if True:
    N_train = int(args.N_all)
    N_test = int(N_train * ratio)
    root = args.root_dir + f'{args.model}' #'./datasets/andi_set/1118_{}-2'.format(args.model)
    os.makedirs(root, exist_ok=True)
    os.makedirs(root + '/video', exist_ok=True)
    os.makedirs(root + '/train', exist_ok=True)
    os.makedirs(root + '/test', exist_ok=True)

    train_set=generate_trajectories(N_train + N_test, T_min, T_max, L, D_max ,num_a_permodel, num_k_permodel,N, [args.K_low, args.K_high], args=args)


    if args.no_sample:
        train_set_new = train_set
    else:
        rand_idx = random.sample([i for i in range(0, N_train - 1)], N_test)
    # print(rand_idx)
    # print(len(train_set))
        test_set = []
        for i in rand_idx:
            # print('processing:', i, end=',')
            test_set.append(train_set[i])
        train_idx = list(range(len(train_set)))
        for i in rand_idx:
            # print('remove:', i, end=',')
            train_idx.remove(i)
        train_set_new = []
        for i in train_idx:
            train_set_new.append(train_set[i])
        print(len(train_set_new))
        
        with open(os.path.join(root, 'test', 'test{}_Nall_{}_T{}_{}_K{}_{}.csv'.format(idx, args.N_all, args.T_low, args.T_high, args.K_low, args.K_high)), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            ttt = test_set
            writer.writerows(ttt)

    # np.save(os.path.join(root, 'train', 'train_{}.npy'.format(idx)))
    # np.save(os.path.join(root, 'test', 'test_{}.npy'.format(idx)))
    with open(os.path.join(root, 'train', 'train{}_Nall_{}_T{}_{}_K{}_{}.csv'.format(idx, args.N_all, args.T_low, args.T_high, args.K_low, args.K_high)), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        ttt = train_set_new
        writer.writerows(ttt)



    idx = idx + per_num

# with open(
#         '{}/2d_train_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(root, N_all_train, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = train_set_new
#     writer.writerows(ttt)


# with open(
#         '{}/2d_test_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(root, N_all_test, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = test_set
#     writer.writerows(ttt)

# N_all_valid = 500
# N=10
# valid_set = generate_trajectories(N_all_valid, T_min, T_max, L, D_max, num_a_permodel, num_k_permodel, N)

# with open(
#         './datasets/andi_set/N5e4_1e2/2d_valid_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(N_all_valid, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = valid_set
#     writer.writerows(ttt)

# N_all_test = 500
# N=10
# test_set = generate_trajectories(N_all_test, T_min, T_max, L, D_max, num_a_permodel, num_k_permodel, N)

# with open(
#         './datasets/andi_set/N5e4_1e2/2d_test_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(N_all_test, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = test_set
#     writer.writerows(ttt)

a=0










#todo:分类网络的数据集我还得想想参数分布怎么设置 还没写好
'''

#分割以后，要进行segment的识别  长度 10-200  包含4个类别 第一个类别 0 iMobile 1 confine 2 diffusion 3 direct  每种都有N条
number_compartments = 50
radius_compartments = 10
compartments_center = models_phenom._distribute_circular_compartments(Nc = number_compartments,
                                                                      r = radius_compartments,
                                                                      L = L # size of the environment
                                                                      )

trajs_model5, labels_model5 = models_phenom().confinement(N = N,
                                                          L = L,
                                                          Ds = [1500*D, 50*D],
                                                          comp_center = compartments_center,
                                                          r = radius_compartments,
                                                          trans = 0.2 # Boundary transmittance
                                                           )

'''

