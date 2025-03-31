import numpy as np    
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from datasets.data import data_factory
from losscompute import get_loss_module
from torch.utils.data import DataLoader
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
import sklearn
from sklearn.metrics import confusion_matrix
# Project modules
from options import Options

from utils import utils

from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from datasets.data import data_factory

# from transmodel import model_factory
from losscompute import get_loss_module
from optimizers import get_optimizer
from utils import utils, analysis
import numpy as np
import pandas as pd
import seaborn as sns

import sys

import matplotlib.pyplot as plt
import os
import traceback
import json
from datetime import datetime
import string
import random
from utils import utils, analysis
from collections import OrderedDict
import time

def normalize(trajs):
    ''' Normalizes trajectories by substracting average and dividing by sqrt of msd
    Arguments:
	- traj: trajectory or ensemble of trajectories to normalize. 
    - dimensions: dimension of the trajectory.
	return: normalized trajectory'''
    if len(trajs.shape) == 1:
        trajs = np.reshape(trajs, (1, trajs.shape[0]))
    trajs = trajs - trajs.mean(axis=1, keepdims=True)
    displacements = (trajs[:,1:] - trajs[:,:-1]).copy()    
    variance = np.std(displacements, axis=1)
    variance[variance == 0] = 1
    new_trajs = np.cumsum((displacements.transpose()/variance).transpose(), axis = 1)
    return np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)

def prepare_dataset(dataset, return_labels=False, n_labels=2, norm=True, shuffle=True, lw=False, to_tensor=True):
    """ Normalizes trajectories from dataset, shuffles them and converts them from np arrays to torch tensors """
    if shuffle: np.random.shuffle(dataset)
    labels = dataset[:,:n_labels]
    dataset = dataset[:,n_labels:]
    if norm: dataset = normalize(dataset)
    if lw: dataset, labels = (dataset[(np.prod((abs(dataset[:,1:])<10e10), axis=1)!=0)], labels[(np.prod((abs(dataset[:,1:])<10e10), axis=1)!=0)])
    if to_tensor: dataset = torch.from_numpy(dataset).reshape(dataset.shape[0],1,dataset.shape[1]).float()
    if return_labels: return dataset, labels
    return dataset

def do_inference(dataset, model, decode=True):
    """ Does inference with given a data set and an autoencoder model 
    return: numpy.array, inferred data set"""
    if isinstance(dataset, np.ndarray):
        raise ValueError('You must convert the data set to a tensor before running inference. Check prepare_dataset')
    model.decode = decode
    return model(dataset).detach().numpy().reshape(dataset.shape[0],-1)

def do_PCA(dataset, model, decode=False, n_components=2):
    """ Does PCA on a data set after inference with an autoencoder model
    return: numpy.array
    """
    inference = do_inference(dataset=dataset, model=model, decode=decode)
    return PCA(n_components=n_components).fit_transform(inference)

def do_TSNE(dataset, model, decode=False, n_components=2):
    """ Does TSNE on a data set after inference with an autoencoder model
    return: numpy.array
    """
    inference = do_inference(dataset=dataset, model=model, decode=decode)
    return TSNE(n_components=n_components).fit_transform(inference)

def do_UMAP(dataset, model, decode=False, n_components=2):
    """ Does UMAP on a data set after inference with an autoencoder model
    return: numpy.array
    """
    inference = do_inference(dataset=dataset, model=model, decode=decode)
    return umap.UMAP(n_components=n_components).fit_transform(inference)

def mrae(prediction, target):
    """ Computes mean relative absolute error between a predictor tensor and a tensor with the target values
    return: float
    """
    return ((prediction-target).abs_()/(target+1)).mean().item()

def swsMSE(prediction, target, scale=1):
    """ Computes the sample-wise scaled MSE between a predictor tensor and a tensor with the target values 
    return: float
    """
    scaling_factor = scale * target.abs().max(dim=-1).values.unsqueeze(-1)
    return F.mse_loss(torch.div(prediction, scaling_factor), torch.div(target, scaling_factor)).item()

def relative_entropy(tensor_a, tensor_b):
    tensor_a = torch.div(tensor_a,sum(tensor_a))
    tensor_b = torch.div(tensor_b,sum(tensor_b))
    return sum(tensor_a*np.log(torch.div(tensor_a,tensor_b)))

"""def eb_parameter(trajectories, msd_ratio=0.6):"""
    
def get_msd(trajectories, msd_ratio=0.6):
    n = round(trajectories.shape[-1]*msd_ratio)
    return np.array([np.mean((trajectories[:,i:]-trajectories[:,:-i])**2, axis=1) for i in range (1,n)]).T

def get_variance(msd):
    return np.mean(np.mean(msd**2, axis=0) - np.mean(msd, axis=0)**2)

class Dataset_Loader(Dataset):
    """ Data loader class for the supervised model """
    def __init__(self, trainset, transform=None):
        self.data = trainset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class swsMSELoss(nn.MSELoss):
    def __init__(self, scale=1):
        super(swsMSELoss, self).__init__()
        self.scale = scale
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        scaling_factor = self.scale * target.abs().max(dim=-1).values.unsqueeze(-1)
        return F.mse_loss(torch.div(input, scaling_factor), torch.div(target, scaling_factor), reduction=self.reduction)

class MAPELoss(_Loss):
    def __init__(self, scale=True, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MAPELoss, self).__init__(size_average, reduce, reduction)
        self.scale = scale

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        scaling_factor = self.scale_tensor(target)
        return torch.div((target-input).abs(),(input.abs()+scaling_factor)).mean()
    
    def scale_tensor(self, target: Tensor):
        if self.scale:
            return target.abs().max(dim=-1).values.unsqueeze(-1)
        return 1


#找出某一类轨迹的索引值集合
#输入：分类任务的groundtruth label，【0，1，2，3，4，5】某一类轨迹， 输出：某一类轨迹的索引值集合
def find_class_order(targets_m,model):
    model_order=[]
    for i, val in enumerate(targets_m):
        if val==model:
            model_order.append(i)
    return model_order

#输入：某一类轨迹的索引值集合
#输出：该类轨迹的regress_groundturth 和 prediction
def from_class_order_find_regression(model_order,targets_a,predictions_a):
    targets_a_model=[]
    prediction_a_model = []
    for i, val in enumerate(model_order):
        targets_a_model.append(targets_a[val])
        prediction_a_model.append(predictions_a[val])
    return targets_a_model,prediction_a_model

def find_thredshold(msp_id,fpr):

    msp=msp_id.cpu().detach().numpy()
    msp=np.sort(msp)
    len_sample=len(msp)
    thredshold_index=int(len_sample*fpr)
    thredshold=msp[thredshold_index]
    return thredshold


def find_thredshold_fpr95(msp_ood, tpr):



    msp = msp_ood.cpu().detach().numpy()
    msp = np.sort(msp)
    len_sample = len(msp)
    thredshold_index = int(len_sample * tpr)+1

    if thredshold_index == len(msp):
        thredshold_index=len(msp)-1

    thredshold = msp[thredshold_index]
    return thredshold

def cal_pre_recall_for_method1(label_all,score_all,thredshold):


    label_pre=[]
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(score_all)):
        if score_all[i]<thredshold:
            label_pre.append(0)   #
        else:
            label_pre.append(1)   #
    for i in range(len(score_all)):
        if label_pre[i]==label_all[i]:
            if label_pre[i] == 1:
                TN += 1
            else:
                TP += 1
        else:
            if label_pre[i] == 1:
                FN += 1
            else:
                FP += 1
    if (TP==0) and (FP==0):
        pre=1
    else:
        pre=TP/(TP+FP)
    recall=TP/(TP+FN)


    return pre,recall




def wrong_ood_id_detection(thredshold,test_indices,test_indices_ood,msp_id,msp_ood,targets_aa,targets_aa_ood,ood_a_real=True,plot=True):

    wrong_ood = 0  # 错误诊断的ood
    wrong_id = 0  # 错误诊断的id
    total_ood = len(test_indices_ood)
    error_ood_a = []
    error_id_a = []
    order_id = []
    order_ood = []
    total_id = len(test_indices)




    ta = targets_aa_ood.cpu().detach().numpy()
    ta.reshape(len(ta), )
    counts_ood_gt, bins_ood_gt, patches_ood_gt = plt.hist(ta, bins=np.arange(0., 2., 0.05))
    plt.close()

    for i in range(len(counts_ood_gt)):
        if counts_ood_gt[i] == 0:
            counts_ood_gt[i] = 1

    for i in range(len(msp_ood)):
        if msp_ood[i] >= thredshold:
            wrong_ood = wrong_ood + 1
            error_ood_a.append(targets_aa_ood[i].item())  #
            order_ood.append(i)

    error_ood = wrong_ood / total_ood
    error_ood_a_gt = np.array(error_ood_a)
    error_ood_a_gt.reshape(len(error_ood_a_gt), 1)

    tpr = 1 - error_ood  #

    counts_ood, bins_ood, patches_ood = plt.hist(error_ood_a_gt, bins=np.arange(0., 2., 0.05))

    # 添加标题和轴标签
    plt.title('Histogram of Array')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示直方图
    plt.close()

    intensity_ood = counts_ood / counts_ood_gt


    if ood_a_real:
        y_a_distribute_ood=np.append(intensity_ood, [intensity_ood[-1]])
        if plot:
            plt.step(bins_ood, y_a_distribute_ood, where='post', lw=3)
            # plt.fill_between(bins_ood, np.append(intensity_ood, [intensity_ood[-1]]), alpha=.5, linewidth=0)
            plt.title('Histogram of wrong ood')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            plt.show()

    gt = targets_aa.cpu().detach().numpy()
    gt.reshape(len(gt), )
    counts_id_gt, bins_id_gt, patches_id_gt = plt.hist(gt, bins=np.arange(0., 2., 0.05))
    plt.close()

    for i in range(len(counts_id_gt)):
        if counts_id_gt[i] == 0:
            counts_id_gt[i] = 1

    for i in range(len(msp_id)):
        if msp_id[i] < thredshold:
            wrong_id = wrong_id + 1
            error_id_a.append(targets_aa[i].item())  #
            order_id.append(i)

    error_id = wrong_id / total_id
    error_id_a_gt = np.array(error_id_a)  #
    error_id_a_gt.reshape(len(error_id_a_gt), 1)

    counts_id, bins_id, patches_id = plt.hist(error_id_a_gt, bins=np.arange(0., 2., 0.05))

    #
    plt.title('Histogram of Array')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    #
    plt.close()

    intensity_id = counts_id / counts_id_gt
    y_a_distribute_id=np.append(intensity_id, [intensity_id[-1]])
    if plot:
        plt.step(bins_id,y_a_distribute_id, where='post', lw=3)
        # plt.fill_between(bins_ood, np.append(intensity_ood, [intensity_ood[-1]]), alpha=.5, linewidth=0)
        plt.title('Histogram of wrong id')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.show()

    fpr=error_id

    if ood_a_real:
        wrong_distribute_plot={'x_ood':bins_ood,'y_ood':y_a_distribute_ood,'x_id':bins_id,'y_id':y_a_distribute_id}

    else:
        wrong_distribute_plot = {'x_id': bins_id,'y_id': y_a_distribute_id}

    return tpr, fpr, order_ood, order_id, wrong_distribute_plot


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


def plot_tsne(features, labels):

    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns

    class_num = len(np.unique(labels))
    latent = features
    tsne_features = tsne.fit_transform(features)  #
    print('tsne_features的shape:', tsne_features.shape)
    '''plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  #
    plt.show()'''

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    fig = plt.figure(figsize=(12, 9), dpi=600)

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df,alpha=0.3).set(title="Bearing data T-SNE projection unsupervised")
    #ax.scatter(df["comp-1"], df["comp-2"], alpha=0.5)  # 设置透明度为0.5
    '''plt.scatter(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    color=sns.color_palette("hls", class_num),
                    data=df).set(title="Bearing data T-SNE projection unsupervised")'''


    plt.show()




class NoFussCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

def show_predic(config,model,test_set1,device,addre):

    data_class = data_factory[config['data_class']]
    dim = config['dimension']
    my_data = data_class(test_set1, dim=dim, n_proc=config['n_proc'],
                         limit_size=config['limit_size'], config=config)  # (sequencelenth*sample_num,feat_dim)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels_m = my_data.labels_dfm.values.flatten()  #
        labels_a = None
    else:
        validation_method = 'ShuffleSplit'
        labels_m = my_data.labels_dfm.values.flatten()  #
        labels_a = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    test_data = data_class(test_set1, dim=dim, n_proc=-1, limit_size=config['limit_size'], config=config)
    test_indices = test_data.all_IDs


    #################################################################################################################################################

    #################################################################################################################################################

    loss_module = get_loss_module(config)


    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    test_dataset = dataset_class(test_data, test_indices)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True,
                             collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    test_evaluator = runner_class(model, test_loader, device, loss_module,
                                  print_interval=config['print_interval'], console=config['console'])
    per_batch = {'target_masks': [], 'targets_m': [], 'targets_a': [], 'predictions_m': [], 'predictions_a': [],
                 'metrics': [], 'IDs': []}
    all_target_prediction = {'targets_m': [], 'targets_a': [], 'predictions_m': [], 'predictions_a': [], 'metrics': []}
    total_samples = 0
    epoch_loss = 0
    epoch_metrics = OrderedDict()
    epoch_num = None
    analyzer = analysis.Analyzer(print_conf_mat=True)
    model = model.eval()

    ###################################################################################################################################
    # 将测试数据载入模型进行测试

    fc_class_weights = None
    fc_reg_weights = None
    for name, param in model.named_parameters():
        if 'output_layer_m.weight' in name:
            fc_class_weights = param.data
        elif 'output_layer_a.weight' in name:
            fc_reg_weights = param.data

    fc = fc_class_weights.cpu().numpy()
    arr = np.full((len(fc) - 1, len(fc) - 1), np.nan)
    arr2 = np.full((len(fc), len(fc)), np.nan)


    for i in range(len(fc)):
        for j in range(len(fc)):
            if i != j:
                w1 = fc[i]
                w2 = fc[j]
                a = np.dot(w1, w2)
                b1 = np.linalg.norm(w1, ord=None, axis=None)
                b2 = np.linalg.norm(w2, ord=None, axis=None)
                arr2[i][j] = a / (b1 * b2)  #

    arr3 = np.full((len(fc)), np.nan)
    for i in range(len(fc)):
        a = arr2[i]
        a[i] = -100000000
        arr3[i] = int(np.argmax(a))
        if a[int(arr3[i])] <= 0:
            arr3[i] = -1  #

    for i, batch in enumerate(test_loader):
        X, targets_m, targets_a, padding_masks, IDs = batch
        targets_m = targets_m.to(device)
        targets_a = targets_a.to(device)
        padding_masks = padding_masks.to(device)  # 0s: ignore
        # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
        ################################################################################################################################
        #

        predictions_m, predictions_a,_,_ = model(X.to(device), padding_masks)

        ################################################################################################################################


        per_batch['targets_m'].append(targets_m.cpu().numpy())
        per_batch['targets_a'].append(targets_a.cpu().numpy())
        per_batch['predictions_m'].append(predictions_m.cpu().detach().numpy())
        per_batch['predictions_a'].append(predictions_a.cpu().detach().numpy())
        #per_batch['metrics'].append([loss.cpu().detach().numpy()])
        per_batch['IDs'].append(IDs)


    if config['task'] == 'classification':
        predictions_m = torch.from_numpy(np.concatenate(per_batch['predictions_m'], axis=0))
        probs = torch.nn.functional.softmax(
            predictions_m)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions_m = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets_m = np.concatenate(per_batch['targets_m'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = analyzer.analyze_classification(predictions_m, targets_m, class_names)

        epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        if model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets_m, probs[:, 1])  # 1D scores needed
            epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets_m, probs[:, 1])
            epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)

        conf_normal = metrics_dict['ConfMatrix_normalized_row']
        conf_real= metrics_dict['ConfMatrix']
        non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)  #
        #non_nan_cols = np.logical_not(np.isnan(conf_normal)).all(axis=0)  #
        NANLine=[]
        for i in range(len(non_nan_rows)):
            if non_nan_rows[i]==False:
                NANLine.append(i)
        '''NANLine_col = []
        for i in range(len(non_nan_cols)):
            if non_nan_cols[i] == False:
                NANLine_col.append(i)'''
        conf_normal=np.delete(conf_normal,NANLine,0)
        conf_real = np.delete(conf_real, NANLine, 0)
        #conf_normal = np.delete(conf_normal, NANLine_col, 1)
        #conf_real = np.delete(conf_real, NANLine_col, 1)
        conf_normal=conf_normal.transpose()
        conf_real=conf_real.transpose()

        fig = plt.figure(figsize=(20, 6), dpi=300)
        #plt.rcParams['figure.figsize'] = (20.0, 10.0)
        # plt.rcParams['figure.dpi'] = 600

        #
        if config['ood_class'] == -1:
            column1 = ['ATTM', 'CTRW','FBM', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM', 'CTRW','FBM', 'LW', 'SBM'])
            index1 = column1

            for i in range(5):
                if np.sum(targets_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            for i in range(5):
                if np.sum(predictions_m==i) ==0:   #
                    new_row = np.zeros((1, conf_normal.shape[1]))  #
                    conf_normal = np.vstack((conf_normal[:i, :], new_row, conf_normal[i:, :]))  #
                    conf_real = np.vstack((conf_real[:i, :], new_row, conf_real[i:, :]))  #

            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                for j in range(conf_normal.shape[1]):
                    aa.iloc[i][j] = 100*data_set1[i][j]

            aa = aa.apply(pd.to_numeric, errors='ignore')  #

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_xlabel('groundtruth', fontsize=16)
            axx.set_ylabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)





        if config['ood_class'] == 0:
            column1 = ['CTRW','FBM', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories=['CTRW','FBM', 'LW', 'SBM'])
            index1 = column1

            for i in range(4):
                if np.sum(targets_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))  #
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_row = np.zeros((1, conf_normal.shape[1]))
                    conf_normal = np.vstack((conf_normal[:i, :], new_row, conf_normal[i:, :]))
                    conf_real = np.vstack((conf_real[:i, :], new_row, conf_real[i:, :]))

            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                for j in range(conf_normal.shape[1]):
                    aa.iloc[i][j] =100* data_set1[i][j]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_xlabel('groundtruth', fontsize=16)
            axx.set_ylabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)


        if config['ood_class'] == 1:
            column1 = ['ATTM','FBM', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM','FBM', 'LW', 'SBM'])
            index1 = column1

            for i in range(4):
                if np.sum(targets_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_row = np.zeros((1, conf_normal.shape[1]))
                    conf_normal = np.vstack((conf_normal[:i, :], new_row, conf_normal[i:, :]))
                    conf_real = np.vstack((conf_real[:i, :], new_row, conf_real[i:, :]))

            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                for j in range(conf_normal.shape[1]):
                    aa.iloc[i][j] = 100*data_set1[i][j]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)


            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_xlabel('groundtruth', fontsize=16)
            axx.set_ylabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)


        if config['ood_class'] == 2:
            column1 = ['ATTM', 'CTRW', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories=['ATTM', 'CTRW', 'LW', 'SBM'])
            index1 = column1

            for i in range(4):
                if np.sum(targets_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_row = np.zeros((1, conf_normal.shape[1]))
                    conf_normal = np.vstack((conf_normal[:i, :], new_row, conf_normal[i:, :]))
                    conf_real = np.vstack((conf_real[:i, :], new_row, conf_real[i:, :]))

            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                for j in range(conf_normal.shape[1]):
                    aa.iloc[i][j] = 100*data_set1[i][j]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_xlabel('groundtruth', fontsize=16)
            axx.set_ylabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)

        if config['ood_class'] == 3:
            column1 = ['ATTM', 'CTRW', 'FBM', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories=['ATTM', 'CTRW', 'FBM', 'SBM'])
            index1 = column1

            for i in range(4):
                if np.sum(targets_m == i) == 0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            for i in range(4):
                if np.sum(predictions_m == i) == 0:
                    new_row = np.zeros((1, conf_normal.shape[1]))
                    conf_normal = np.vstack((conf_normal[:i, :], new_row, conf_normal[i:, :]))
                    conf_real = np.vstack((conf_real[:i, :], new_row, conf_real[i:, :]))

            metrics_dict['ConfMatrix_normalized_row'] = conf_normal
            metrics_dict['ConfMatrix'] = conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                for j in range(conf_normal.shape[1]):
                    aa.iloc[i][j] =100* data_set1[i][j]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax3)

            # axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_xlabel('groundtruth', fontsize=16)
            axx.set_ylabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)

        if config['ood_class'] == 4:
            column1 = ['ATTM', 'CTRW','FBM', 'LW']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM', 'CTRW','FBM', 'LW'])
            index1 = column1

            for i in range(4):
                if np.sum(targets_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_row = np.zeros((1, conf_normal.shape[1]))
                    conf_normal = np.vstack((conf_normal[:i, :], new_row, conf_normal[i:, :]))
                    conf_real = np.vstack((conf_real[:i, :], new_row, conf_real[i:, :]))

            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                for j in range(conf_normal.shape[1]):
                    aa.iloc[i][j] = 100*data_set1[i][j]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_xlabel('groundtruth', fontsize=16)
            axx.set_ylabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)
        class_plot=aa

    from sklearn.metrics import f1_score

    f1 = f1_score(targets_m, predictions_m, average='micro')
    # print(f1)


    #########################################################################################################
    load_reg = addre + 'checkpoints/model_best_reg.pth'
    if (os.path.isfile(load_reg)):
        # config['load_model'] = 'output/_2022-12-30_21-35-30_g4L/checkpoints/model_best.pth'
        config['change_output'] = False
        # Initialize optimizer

        if config['global_reg']:
            weight_decay = config['l2_reg']
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config['l2_reg']

        optim_class = get_optimizer(config['optimizer'])
        optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
        model_reg, optimizer, start_epoch = utils.load_model(model, load_reg, optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
        # model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model_reg = model_reg.to(device)
        per_batch = {'target_masks': [], 'targets_m': [], 'targets_a': [], 'predictions_m': [], 'predictions_a': [],
                     'metrics': [], 'IDs': []}

        for i, batch in enumerate(test_loader):
            X, targets_m, targets_a, padding_masks, IDs = batch
            targets_m = targets_m.to(device)
            targets_a = targets_a.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
            ################################################################################################################################
            # 输出logits

            predictions_m, predictions_a,_,_ = model_reg(X.to(device), padding_masks)

            ###############################################################################################################################
            per_batch['targets_m'].append(targets_m.cpu().numpy())
            per_batch['targets_a'].append(targets_a.cpu().numpy())
            per_batch['predictions_m'].append(predictions_m.cpu().detach().numpy())
            per_batch['predictions_a'].append(predictions_a.cpu().detach().numpy())
            #per_batch['metrics'].append([loss.cpu().detach().numpy()])
            per_batch['IDs'].append(IDs)

        #################################################################################################################################






    from sklearn.metrics import mean_squared_error, mean_absolute_error

    targets_a = per_batch['targets_a']
    targets_m = per_batch['targets_m']
    predictions_m = per_batch['predictions_m']
    predictions_a = per_batch['predictions_a']

    targets_m = np.concatenate(targets_m, axis=0).flatten()
    targets_a = np.concatenate(targets_a, axis=0).flatten()

    predictions_a = np.concatenate(predictions_a, axis=0).flatten()
    predictions_m = torch.from_numpy(np.concatenate(predictions_m, axis=0))
    probs = torch.nn.functional.softmax(
        predictions_m)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions_m = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    total_rmse = np.sqrt(mean_squared_error(targets_a, predictions_a))
    #print(total_rmse)
    total_mae = mean_absolute_error(targets_a, predictions_a)
    #print(total_mae)

    if config['ood_class'] == -1:
        attm = 0
        ctrw = 1
        fbm = 2
        lw = 3
        sbm = 4

        model_order_attm = find_class_order(targets_m, attm)
        model_order_ctrw = find_class_order(targets_m, ctrw)
        model_order_fbm = find_class_order(targets_m, fbm)
        model_order_lw = find_class_order(targets_m, lw)
        model_order_sbm = find_class_order(targets_m, sbm)

        targets_a_attm, predictions_a_attm = from_class_order_find_regression(model_order_attm, targets_a,
                                                                                     predictions_a)
        if len(targets_a_attm) == 0:
            targets_a_attm = [1]
            predictions_a_attm = [1]

        targets_a_ctrw, predictions_a_ctrw = from_class_order_find_regression(model_order_ctrw, targets_a,
                                                                              predictions_a)
        if len(targets_a_ctrw) == 0:
            targets_a_ctrw = [1]
            predictions_a_ctrw = [1]

        targets_a_fbm, predictions_a_fbm = from_class_order_find_regression(model_order_fbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_fbm) == 0:
            targets_a_fbm = [1]
            predictions_a_fbm = [1]

        targets_a_lw, predictions_a_lw = from_class_order_find_regression(model_order_lw, targets_a,
                                                                          predictions_a)
        if len(targets_a_lw) == 0:
            targets_a_lw = [1]
            predictions_a_lw = [1]

        targets_a_sbm, predictions_a_sbm = from_class_order_find_regression(model_order_sbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_sbm) == 0:
            targets_a_sbm = [1]
            predictions_a_sbm = [1]

        attm_rmse = np.sqrt(mean_squared_error(targets_a_attm, predictions_a_attm))
        attm_mae = mean_absolute_error(targets_a_attm, predictions_a_attm)

        ctrw_rmse = np.sqrt(mean_squared_error(targets_a_ctrw, predictions_a_ctrw))
        ctrw_mae = mean_absolute_error(targets_a_ctrw, predictions_a_ctrw)

        fbm_rmse = np.sqrt(mean_squared_error(targets_a_fbm, predictions_a_fbm))
        fbm_mae = mean_absolute_error(targets_a_fbm, predictions_a_fbm)

        lw_rmse = np.sqrt(mean_squared_error(targets_a_lw, predictions_a_lw))
        lw_mae = mean_absolute_error(targets_a_lw, predictions_a_lw)

        sbm_rmse = np.sqrt(mean_squared_error(targets_a_sbm, predictions_a_sbm))
        sbm_mae = mean_absolute_error(targets_a_sbm, predictions_a_sbm)

        x = [1, 2, 3, 4, 5]
        y = [attm_mae, ctrw_mae, fbm_mae, lw_mae, sbm_mae]
        c = total_mae

        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["ATTM","CTRW", "FBM", "LW", "SBM"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")
        mae_plot = {'x': x, 'y': y, 'total_mae': c, 'tick_label': ["ATTM","CTRW", "FBM", "LW", "SBM"]}





    elif config['ood_class'] == 0:
        ctrw = 0
        fbm = 1
        lw = 2
        sbm = 3

        model_order_ctrw = find_class_order(targets_m, ctrw)
        model_order_fbm = find_class_order(targets_m, fbm)
        model_order_lw = find_class_order(targets_m, lw)
        model_order_sbm = find_class_order(targets_m, sbm)

        targets_a_ctrw, predictions_a_ctrw = from_class_order_find_regression(model_order_ctrw, targets_a,
                                                                                     predictions_a)
        if len(targets_a_ctrw) == 0:
            targets_a_ctrw=[1]
            predictions_a_ctrw=[1]

        targets_a_fbm, predictions_a_fbm = from_class_order_find_regression(model_order_fbm, targets_a,
                                                                                   predictions_a)
        if len(targets_a_fbm) == 0:
            targets_a_fbm=[1]
            predictions_a_fbm=[1]

        targets_a_lw, predictions_a_lw = from_class_order_find_regression(model_order_lw, targets_a,
                                                                                 predictions_a)
        if len(targets_a_lw) == 0:
            targets_a_lw=[1]
            predictions_a_lw=[1]


        targets_a_sbm, predictions_a_sbm = from_class_order_find_regression(model_order_sbm, targets_a,
                                                                                   predictions_a)
        if len(targets_a_sbm) == 0:
            targets_a_sbm=[1]
            predictions_a_sbm=[1]

        ctrw_rmse = np.sqrt(mean_squared_error(targets_a_ctrw, predictions_a_ctrw))
        ctrw_mae = mean_absolute_error(targets_a_ctrw, predictions_a_ctrw)

        fbm_rmse = np.sqrt(mean_squared_error(targets_a_fbm, predictions_a_fbm))
        fbm_mae = mean_absolute_error(targets_a_fbm, predictions_a_fbm)


        lw_rmse = np.sqrt(mean_squared_error(targets_a_lw, predictions_a_lw))
        lw_mae = mean_absolute_error(targets_a_lw, predictions_a_lw)

        sbm_rmse = np.sqrt(mean_squared_error(targets_a_sbm, predictions_a_sbm))
        sbm_mae = mean_absolute_error(targets_a_sbm, predictions_a_sbm)

        x = [1, 2, 3, 4]
        y = [ctrw_mae, fbm_mae, lw_mae, sbm_mae]
        c = total_mae

        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["CTRW", "FBM", "LW", "SBM"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")
        mae_plot = {'x': x, 'y': y, 'total_mae': c, 'tick_label': ["CTRW", "FBM", "LW", "SBM"]}

    elif config['ood_class'] == 1:
        attm = 0
        fbm = 1
        lw = 2
        sbm = 3

        model_order_attm = find_class_order(targets_m, attm)
        model_order_fbm = find_class_order(targets_m, fbm)
        model_order_lw = find_class_order(targets_m, lw)
        model_order_sbm = find_class_order(targets_m, sbm)

        targets_a_attm, predictions_a_attm = from_class_order_find_regression(model_order_attm, targets_a,
                                                                              predictions_a)
        if len(targets_a_attm) == 0:
            targets_a_attm = [1]
            predictions_a_attm = [1]


        targets_a_fbm, predictions_a_fbm = from_class_order_find_regression(model_order_fbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_fbm) == 0:
            targets_a_fbm = [1]
            predictions_a_fbm = [1]

        targets_a_lw, predictions_a_lw = from_class_order_find_regression(model_order_lw, targets_a,
                                                                          predictions_a)
        if len(targets_a_lw) == 0:
            targets_a_lw = [1]
            predictions_a_lw = [1]

        targets_a_sbm, predictions_a_sbm = from_class_order_find_regression(model_order_sbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_sbm) == 0:
            targets_a_sbm = [1]
            predictions_a_sbm = [1]

        attm_rmse = np.sqrt(mean_squared_error(targets_a_attm, predictions_a_attm))
        attm_mae = mean_absolute_error(targets_a_attm, predictions_a_attm)

        fbm_rmse = np.sqrt(mean_squared_error(targets_a_fbm, predictions_a_fbm))
        fbm_mae = mean_absolute_error(targets_a_fbm, predictions_a_fbm)

        lw_rmse = np.sqrt(mean_squared_error(targets_a_lw, predictions_a_lw))
        lw_mae = mean_absolute_error(targets_a_lw, predictions_a_lw)

        sbm_rmse = np.sqrt(mean_squared_error(targets_a_sbm, predictions_a_sbm))
        sbm_mae = mean_absolute_error(targets_a_sbm, predictions_a_sbm)

        x = [1, 2, 3, 4]
        y = [attm_mae, fbm_mae, lw_mae, sbm_mae]
        c = total_mae

        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["ATTM", "FBM", "LW", "SBM"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")
        mae_plot = {'x': x, 'y': y, 'total_mae': c, 'tick_label': ["ATTM", "FBM", "LW", "SBM"]}

    elif config['ood_class'] == 2:
        attm = 0
        ctrw = 1
        lw = 2
        sbm = 3

        model_order_attm = find_class_order(targets_m, attm)
        model_order_ctrw = find_class_order(targets_m, ctrw)
        model_order_lw = find_class_order(targets_m, lw)
        model_order_sbm = find_class_order(targets_m, sbm)

        targets_a_attm, predictions_a_attm = from_class_order_find_regression(model_order_attm, targets_a,
                                                                              predictions_a)
        if len(targets_a_attm) == 0:
            targets_a_attm = [1]
            predictions_a_attm = [1]

        targets_a_ctrw, predictions_a_ctrw = from_class_order_find_regression(model_order_ctrw, targets_a,
                                                                              predictions_a)
        if len(targets_a_ctrw) == 0:
            targets_a_ctrw = [1]
            predictions_a_ctrw = [1]

        targets_a_lw, predictions_a_lw = from_class_order_find_regression(model_order_lw, targets_a,
                                                                          predictions_a)
        if len(targets_a_lw) == 0:
            targets_a_lw = [1]
            predictions_a_lw = [1]

        targets_a_sbm, predictions_a_sbm = from_class_order_find_regression(model_order_sbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_sbm) == 0:
            targets_a_sbm = [1]
            predictions_a_sbm = [1]

        attm_rmse = np.sqrt(mean_squared_error(targets_a_attm, predictions_a_attm))
        attm_mae = mean_absolute_error(targets_a_attm, predictions_a_attm)

        ctrw_rmse = np.sqrt(mean_squared_error(targets_a_ctrw, predictions_a_ctrw))
        ctrw_mae = mean_absolute_error(targets_a_ctrw, predictions_a_ctrw)

        lw_rmse = np.sqrt(mean_squared_error(targets_a_lw, predictions_a_lw))
        lw_mae = mean_absolute_error(targets_a_lw, predictions_a_lw)

        sbm_rmse = np.sqrt(mean_squared_error(targets_a_sbm, predictions_a_sbm))
        sbm_mae = mean_absolute_error(targets_a_sbm, predictions_a_sbm)

        x = [1, 2, 3, 4]
        y = [attm_mae, ctrw_mae, lw_mae, sbm_mae]
        c = total_mae

        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["ATTM", "CTRW", "LW", "SBM"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")
        mae_plot = {'x': x, 'y': y, 'total_mae': c, 'tick_label': ["ATTM", "CTRW", "LW", "SBM"]}

    elif config['ood_class'] == 3:
        attm = 0
        ctrw = 1
        fbm = 2
        sbm = 3

        model_order_attm =find_class_order(targets_m, attm)
        model_order_ctrw = find_class_order(targets_m, ctrw)
        model_order_fbm = find_class_order(targets_m, fbm)
        model_order_sbm = find_class_order(targets_m, sbm)

        targets_a_attm, predictions_a_attm = from_class_order_find_regression(model_order_attm, targets_a,
                                                                              predictions_a)
        if len(targets_a_attm) == 0:
            targets_a_attm = [1]
            predictions_a_attm = [1]

        targets_a_ctrw, predictions_a_ctrw = from_class_order_find_regression(model_order_ctrw, targets_a,
                                                                              predictions_a)
        if len(targets_a_ctrw) == 0:
            targets_a_ctrw = [1]
            predictions_a_ctrw = [1]

        targets_a_fbm, predictions_a_fbm = from_class_order_find_regression(model_order_fbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_fbm) == 0:
            targets_a_fbm = [1]
            predictions_a_fbm = [1]


        targets_a_sbm, predictions_a_sbm = from_class_order_find_regression(model_order_sbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_sbm) == 0:
            targets_a_sbm = [1]
            predictions_a_sbm = [1]

        attm_rmse = np.sqrt(mean_squared_error(targets_a_attm, predictions_a_attm))
        attm_mae = mean_absolute_error(targets_a_attm, predictions_a_attm)

        ctrw_rmse = np.sqrt(mean_squared_error(targets_a_ctrw, predictions_a_ctrw))
        ctrw_mae = mean_absolute_error(targets_a_ctrw, predictions_a_ctrw)

        fbm_rmse = np.sqrt(mean_squared_error(targets_a_fbm, predictions_a_fbm))
        fbm_mae = mean_absolute_error(targets_a_fbm, predictions_a_fbm)

        sbm_rmse = np.sqrt(mean_squared_error(targets_a_sbm, predictions_a_sbm))
        sbm_mae = mean_absolute_error(targets_a_sbm, predictions_a_sbm)

        x = [1, 2, 3, 4]
        y = [attm_mae, ctrw_mae, fbm_mae, sbm_mae]
        c = total_mae

        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["ATTM", "CTRW", "FBM", "SBM"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")
        mae_plot = {'x': x, 'y': y, 'total_mae': c, 'tick_label': ["ATTM", "CTRW", "FBM", "SBM"]}

    else:
        attm = 0
        ctrw = 1
        fbm = 2
        lw = 3

        model_order_attm =find_class_order(targets_m, attm)
        model_order_ctrw = find_class_order(targets_m, ctrw)
        model_order_fbm =find_class_order(targets_m, fbm)
        model_order_lw = find_class_order(targets_m, lw)

        targets_a_attm, predictions_a_attm = from_class_order_find_regression(model_order_attm, targets_a,
                                                                              predictions_a)
        if len(targets_a_attm) == 0:
            targets_a_attm = [1]
            predictions_a_attm = [1]

        targets_a_ctrw, predictions_a_ctrw = from_class_order_find_regression(model_order_ctrw, targets_a,
                                                                              predictions_a)
        if len(targets_a_ctrw) == 0:
            targets_a_ctrw = [1]
            predictions_a_ctrw = [1]

        targets_a_fbm, predictions_a_fbm = from_class_order_find_regression(model_order_fbm, targets_a,
                                                                            predictions_a)
        if len(targets_a_fbm) == 0:
            targets_a_fbm = [1]
            predictions_a_fbm = [1]

        targets_a_lw, predictions_a_lw = from_class_order_find_regression(model_order_lw, targets_a,
                                                                          predictions_a)
        if len(targets_a_lw) == 0:
            targets_a_lw = [1]
            predictions_a_lw = [1]


        attm_rmse = np.sqrt(mean_squared_error(targets_a_attm, predictions_a_attm))
        attm_mae = mean_absolute_error(targets_a_attm, predictions_a_attm)

        ctrw_rmse = np.sqrt(mean_squared_error(targets_a_ctrw, predictions_a_ctrw))
        ctrw_mae = mean_absolute_error(targets_a_ctrw, predictions_a_ctrw)

        fbm_rmse = np.sqrt(mean_squared_error(targets_a_fbm, predictions_a_fbm))
        fbm_mae = mean_absolute_error(targets_a_fbm, predictions_a_fbm)

        lw_rmse = np.sqrt(mean_squared_error(targets_a_lw, predictions_a_lw))
        lw_mae = mean_absolute_error(targets_a_lw, predictions_a_lw)

        x = [1, 2, 3, 4]
        y = [attm_mae, ctrw_mae, fbm_mae, lw_mae]
        c = total_mae

        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["ATTM", "CTRW", "FBM", "LW"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")

        mae_plot = {'x': x, 'y': y, 'total_mae': c,'tick_label':["ATTM", "CTRW", "FBM", "LW"]}

    aa = pd.DataFrame(index=range(3), columns=['a', 'b', 'c'])
    step = 0.1
    alen = np.arange(0, 2.00, step)
    F = len(alen)
    data_set1 = np.random.rand(F, F)
    # FF = len(targets_m)
    FF = len(targets_a)
    simu = np.random.randint(100, size=(F, F))
    for z in range(FF):
        if predictions_a[z] > 2:
            predictions_a[z] = 2
        if predictions_a[z] < 0:
            predictions_a[z] = 0.0001

    for j in range(F):
        for i in range(F):
            simu[i][j] = 0
            a_target_low = j * step
            a_target_high = (j + 1) * step
            a_pred_low = i * step
            a_pred_high = (i + 1) * step
            for z in range(FF):
                if ((targets_a[z] > a_target_low) and (targets_a[z] <= a_target_high)):
                    if ((predictions_a[z] > a_pred_low) and (predictions_a[z] <= a_pred_high)):
                        simu[i][j] = simu[i][j] + 1
                    # elif (predictions_a[z] >2):

    ttt = 0
    for j in range(F):
        for i in range(F):
            ttt = ttt + simu[i][j]
    #print(ttt)

    for i in range(F):
        ss = sum(simu[:, i])
        for j in range(F):
            if ss==0:
                data_set1[j, i] =0
            else:
                data_set1[j, i] = float(simu[j, i]) / float(ss)

        # simu[:,i]=simu[:,i]/ss

    # data_set1 = simu

    index1 = []
    for i in range(F):
        if alen[i] == 0.5:
            index1.append('0.5')
        elif alen[i] == 1:
            index1.append('1.0')
        elif alen[i] == 1.5:
            index1.append('1.5')
        else:
            index1.append('')
    column1 = index1

    column1 = pd.CategoricalIndex(index1, ordered=False, categories=["", "0.5", "1.0", "1.5"])
    index1 = column1
    aa = pd.DataFrame(index=index1, columns=column1)

    data_set1 = np.array(data_set1, dtype=np.float)

    for i in range(F):
        aa.iloc[i] = data_set1[i]
        # aa.iloc[i]=aa.iloc[i].astype('float')

    aa = aa.apply(pd.to_numeric, errors='ignore')

    sns.set_context({"figure.figsize": (8, 8)})

    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)

    #axx = sns.heatmap(data=aa, square=True, cmap='coolwarm', vmin=-0.6, vmax=0.4, cbar=False, ax=ax3)
    axx = sns.heatmap(data=aa, square=True, cmap='Blues', cbar=False, ax=ax3)

    axx.invert_yaxis()
    plt.setp(axx.get_xticklabels(), fontsize=16)
    plt.setp(axx.get_yticklabels(), fontsize=16)
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='x', labelsize=20)
    # axx.set_xticks(fontsize=20)
    # axx.set_yticks(fontsize=20)
    axx.set_xlabel('groundtruth', fontsize=16)
    axx.set_ylabel('prediction', fontsize=16)
    axx.set_title("Regression Heat Map", fontsize=16)

    heat_a_plot=aa
    plt.show()

    #######型##################################################################################################
    load = addre + 'checkpoints/model_best.pth'
    if (os.path.isfile(load_reg)):
        # config['load_model'] = 'output/_2022-12-30_21-35-30_g4L/checkpoints/model_best.pth'
        config['change_output'] = False
        # Initialize optimizer
        # config['l2_reg']
        # config['global_reg']
        if config['global_reg']:
            weight_decay = config['l2_reg']
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config['l2_reg']
        # config['optimizer']选择优化器
        optim_class = get_optimizer(config['optimizer'])
        optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
        model, optimizer, start_epoch = utils.load_model(model, load, optimizer, config['resume'],
                                                             config['change_output'],
                                                             config['lr'],
                                                             config['lr_step'],
                                                             config['lr_factor'])
        # model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model = model.to(device)




    return test_loader,f1,total_mae,metrics_dict,class_plot,mae_plot,heat_a_plot


def show_predic_ood(config,model,test_set1,device,addre,filep_ood):

    data_class = data_factory[config['data_class']]
    dim = config['dimension']
    my_data = data_class(test_set1, dim=dim, n_proc=config['n_proc'],
                         limit_size=config['limit_size'], config=config)  # (sequencelenth*sample_num,feat_dim)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels_m = my_data.labels_dfm.values.flatten()
        labels_a = None
    else:
        validation_method = 'ShuffleSplit'
        labels_m = my_data.labels_dfm.values.flatten()
        labels_a = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    test_data = data_class(test_set1, dim=dim, n_proc=-1, limit_size=config['limit_size'], config=config)
    test_indices = test_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0


    #################################################################################################################################################
    ##加载模型
    #################################################################################################################################################

    loss_module = get_loss_module(config)


    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    test_dataset = dataset_class(test_data, test_indices)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True,
                             collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    test_evaluator = runner_class(model, test_loader, device, loss_module,
                                  print_interval=config['print_interval'], console=config['console'])
    per_batch = {'target_masks': [], 'targets_m': [], 'targets_a': [], 'predictions_m': [], 'predictions_a': [],
                 'metrics': [], 'IDs': []}
    all_target_prediction = {'targets_m': [], 'targets_a': [], 'predictions_m': [], 'predictions_a': [], 'metrics': []}
    total_samples = 0
    epoch_loss = 0
    epoch_metrics = OrderedDict()
    epoch_num = None
    analyzer = analysis.Analyzer(print_conf_mat=True)
    model = model.eval()

    ###################################################################################################################################

    fc_class_weights = None
    fc_reg_weights = None
    for name, param in model.named_parameters():
        if 'output_layer_m.weight' in name:
            fc_class_weights = param.data
        elif 'output_layer_a.weight' in name:
            fc_reg_weights = param.data

    fc = fc_class_weights.cpu().numpy()
    arr = np.full((len(fc) - 1, len(fc) - 1), np.nan)
    arr2 = np.full((len(fc), len(fc)), np.nan)


    for i in range(len(fc)):
        for j in range(len(fc)):
            if i != j:
                w1 = fc[i]
                w2 = fc[j]
                a = np.dot(w1, w2)
                b1 = np.linalg.norm(w1, ord=None, axis=None)
                b2 = np.linalg.norm(w2, ord=None, axis=None)
                arr2[i][j] = a / (b1 * b2)

    arr3 = np.full((len(fc)), np.nan)
    for i in range(len(fc)):
        a = arr2[i]
        a[i] = -100000000
        arr3[i] = int(np.argmax(a))
        if a[int(arr3[i])] <= 0:
            arr3[i] = -1

    for i, batch in enumerate(test_loader):
        X, targets_m, targets_a, padding_masks, IDs = batch
        targets_m = targets_m.to(device)
        targets_a = targets_a.to(device)
        padding_masks = padding_masks.to(device)  # 0s: ignore
        # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
        ################################################################################################################################


        predictions_m, predictions_a,_,_ = model(X.to(device), padding_masks)

        ################################################################################################################################

        per_batch['targets_m'].append(targets_m.cpu().numpy())
        per_batch['targets_a'].append(targets_a.cpu().numpy())
        per_batch['predictions_m'].append(predictions_m.cpu().detach().numpy())
        per_batch['predictions_a'].append(predictions_a.cpu().detach().numpy())
        #per_batch['metrics'].append([loss.cpu().detach().numpy()])
        per_batch['IDs'].append(IDs)


    if config['task'] == 'classification':
        predictions_m = torch.from_numpy(np.concatenate(per_batch['predictions_m'], axis=0))
        probs = torch.nn.functional.softmax(
            predictions_m)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions_m = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets_m = np.concatenate(per_batch['targets_m'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until decide how to pass class names
        if config['ood_class'] ==4:
            for i in range(len(targets_m)):
                targets_m[i]=targets_m[i]-1
        if config['ood_class'] == -1:
            for i in range(len(targets_m)):
                targets_m[i] = targets_m[i] - 1


        metrics_dict = analyzer.analyze_classification(predictions_m, targets_m, class_names)

        epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        if model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets_m, probs[:, 1])  # 1D scores needed
            epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets_m, probs[:, 1])
            epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)

        conf_normal= metrics_dict['ConfMatrix_normalized_row']

        conf_real = metrics_dict['ConfMatrix']
        #conf_normal = conf_normal.transpose()
        #conf_real = conf_real.transpose()

        fig = plt.figure(figsize=(20, 6), dpi=300)
        # plt.rcParams['figure.figsize'] = (20.0, 10.0)
        # plt.rcParams['figure.dpi'] = 600


        if config['ood_class'] == -1:
            column1 = ['ATTM', 'CTRW','FBM', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM', 'CTRW','FBM', 'LW', 'SBM'])
            index1 = ['OOD']
            index1 = pd.CategoricalIndex(index1, ordered=False, categories=['OOD'])
            for i in range(5):
                if (np.sum(predictions_m == i) == 0):
                    if i < len(conf_normal[0]):
                        if conf_normal[-1][i]!=0:
                            new_col = np.zeros((conf_normal.shape[0], 1))
                            conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                            conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))
                    else:
                        new_col = np.zeros((conf_normal.shape[0], 1))
                        conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                        conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)

            for i in range(len(non_nan_rows)):
                if non_nan_rows[i]:
                    oodline=i

            conf_normal=conf_normal[oodline]
            conf_real=conf_real[oodline]
            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                aa.iloc[0][i] = 100*data_set1[i]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_ylabel('groundtruth', fontsize=16)
            axx.set_xlabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)

        if config['ood_class'] == 0:
            column1 = [ 'CTRW', 'FBM', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories=['CTRW', 'FBM', 'LW', 'SBM'])
            index1 = ['ATTM']
            index1 = pd.CategoricalIndex(index1, ordered=False, categories=['ATTM'])
            for i in range(4):
                if np.sum(predictions_m == i) == 0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)

            for i in range(len(non_nan_rows)):
                if non_nan_rows[i]:
                    oodline = i

            conf_normal = conf_normal[oodline]
            conf_real = conf_real[oodline]
            metrics_dict['ConfMatrix_normalized_row'] = conf_normal
            metrics_dict['ConfMatrix'] = conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                aa.iloc[0][i] = 100*data_set1[i]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax3)

            # axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_ylabel('groundtruth', fontsize=16)
            axx.set_xlabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)

        if config['ood_class'] == 1:
            column1 = ['ATTM','FBM', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM','FBM', 'LW', 'SBM'])
            index1 = ['CTRW']
            index1 = pd.CategoricalIndex(index1, ordered=False, categories=['CTRW'])
            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)

            for i in range(len(non_nan_rows)):
                if non_nan_rows[i]:
                    oodline=i

            conf_normal=conf_normal[oodline]
            conf_real=conf_real[oodline]
            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                aa.iloc[0][i] = 100*data_set1[i]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_ylabel('groundtruth', fontsize=16)
            axx.set_xlabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)


        if config['ood_class'] == 2:
            column1 = ['ATTM', 'CTRW', 'LW', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM', 'CTRW', 'LW', 'SBM'])
            index1 = ['FBM']
            index1 = pd.CategoricalIndex(index1, ordered=False, categories=['FBM'])
            if 'oodclass_2_without_outlier' in filep_ood:
                for i in range(4):
                    if np.sum(predictions_m==i) ==0:
                        new_col = np.zeros((conf_normal.shape[0], 1))
                        conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                        conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)

            for i in range(len(non_nan_rows)):
                if non_nan_rows[i]:
                    oodline=i

            conf_normal=conf_normal[oodline]
            conf_real=conf_real[oodline]
            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                aa.iloc[0][i] = 100*data_set1[i]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_ylabel('groundtruth', fontsize=16)
            axx.set_xlabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)

        if config['ood_class'] == 3:
            column1 = ['ATTM', 'CTRW','FBM', 'SBM']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM', 'CTRW','FBM', 'SBM'])
            index1 = ['LW']
            index1 = pd.CategoricalIndex(index1, ordered=False, categories=['LW'])
            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)

            for i in range(len(non_nan_rows)):
                if non_nan_rows[i]:
                    oodline=i

            conf_normal=conf_normal[oodline]
            conf_real=conf_real[oodline]
            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                aa.iloc[0][i] = 100*data_set1[i]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_ylabel('groundtruth', fontsize=16)
            axx.set_xlabel('prediction', fontsize=16)
            axx.set_title("Classification metrics", fontsize=16)


        if config['ood_class'] == 4:
            column1 = ['ATTM', 'CTRW','FBM', 'LW']
            column1 = pd.CategoricalIndex(column1, ordered=False, categories= ['ATTM', 'CTRW','FBM', 'LW'])
            index1 = ['SBM']
            index1 = pd.CategoricalIndex(index1, ordered=False, categories=['SBM'])
            for i in range(4):
                if np.sum(predictions_m==i) ==0:
                    new_col = np.zeros((conf_normal.shape[0], 1))
                    conf_normal = np.hstack((conf_normal[:, :i], new_col, conf_normal[:, i:]))
                    conf_real = np.hstack((conf_real[:, :i], new_col, conf_real[:, i:]))

            non_nan_rows = np.logical_not(np.isnan(conf_normal)).all(axis=1)

            for i in range(len(non_nan_rows)):
                if non_nan_rows[i]:
                    oodline=i

            conf_normal=conf_normal[oodline]
            conf_real=conf_real[oodline]
            metrics_dict['ConfMatrix_normalized_row']=conf_normal
            metrics_dict['ConfMatrix']=conf_real

            aa = pd.DataFrame(index=index1, columns=column1)
            data_set1 = np.array(conf_normal, dtype=np.float)

            for i in range(conf_normal.shape[0]):
                aa.iloc[0][i] = 100*data_set1[i]

            aa = aa.apply(pd.to_numeric, errors='ignore')

            sns.set_context({"figure.figsize": (8, 8)})

            ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

            axx = sns.heatmap(data=aa, square=True, annot=True,fmt='.2f' ,cmap='Blues', cbar=False,ax=ax3)

            #axx.invert_yaxis()
            plt.setp(axx.get_xticklabels(), fontsize=16)
            plt.setp(axx.get_yticklabels(), fontsize=16)
            axx.set_ylabel('groundtruth', fontsize=16)
            axx.set_xlabel('prediction', fontsize=16)
            axx.set_title("Classification metrics",fontsize=16)


        class_plot=aa


    from sklearn.metrics import f1_score

    f1 = f1_score(targets_m, predictions_m, average='micro')
    # print(f1)


    #########################################################################################################
    load_reg = addre + 'checkpoints/model_best_reg.pth'

    if (os.path.isfile(load_reg) ):
        # config['load_model'] = 'output/_2022-12-30_21-35-30_g4L/checkpoints/model_best.pth'
        config['change_output'] = False
        if config['global_reg']:
            weight_decay = config['l2_reg']
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config['l2_reg']
        # config['optimizer']选择优化器
        optim_class = get_optimizer(config['optimizer'])
        optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
        model_reg, optimizer, start_epoch = utils.load_model(model, load_reg, optimizer, config['resume'],
                                                             config['change_output'],
                                                             config['lr'],
                                                             config['lr_step'],
                                                             config['lr_factor'])

        # model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model_reg = model_reg.to(device)

        per_batch = {'target_masks': [], 'targets_m': [], 'targets_a': [], 'predictions_m': [], 'predictions_a': [],
                     'metrics': [], 'IDs': []}

        for i, batch in enumerate(test_loader):
            X, targets_m, targets_a, padding_masks, IDs = batch
            targets_m = targets_m.to(device)
            targets_a = targets_a.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
            ################################################################################################################################
            # 输出logits

            predictions_m, predictions_a,_,_ = model_reg(X.to(device), padding_masks)

            ################################################################################################################################

            per_batch['targets_m'].append(targets_m.cpu().numpy())
            per_batch['targets_a'].append(targets_a.cpu().numpy())
            per_batch['predictions_m'].append(predictions_m.cpu().detach().numpy())
            per_batch['predictions_a'].append(predictions_a.cpu().detach().numpy())
            #per_batch['metrics'].append([loss.cpu().detach().numpy()])
            per_batch['IDs'].append(IDs)
    ############################################################################################################################



    from sklearn.metrics import mean_squared_error, mean_absolute_error

    targets_a = per_batch['targets_a']
    targets_m = per_batch['targets_m']
    predictions_m = per_batch['predictions_m']
    predictions_a = per_batch['predictions_a']

    targets_m = np.concatenate(targets_m, axis=0).flatten()
    targets_a = np.concatenate(targets_a, axis=0).flatten()

    predictions_a = np.concatenate(predictions_a, axis=0).flatten()
    predictions_m = torch.from_numpy(np.concatenate(predictions_m, axis=0))
    probs = torch.nn.functional.softmax(
        predictions_m)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions_m = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample

    total_rmse = np.sqrt(mean_squared_error(targets_a, predictions_a))
    #print(total_rmse)
    total_mae = mean_absolute_error(targets_a, predictions_a)
    #print(total_mae)


    attm = 0
    ctrw = 1
    fbm = 2
    lw = 3
    sbm = 4

    model_order_attm = find_class_order(targets_m, attm)
    model_order_ctrw = find_class_order(targets_m, ctrw)
    model_order_fbm = find_class_order(targets_m, fbm)
    model_order_lw = find_class_order(targets_m, lw)
    model_order_sbm = find_class_order(targets_m, sbm)

    targets_a_attm, predictions_a_attm = from_class_order_find_regression(model_order_attm, targets_a,
                                                                                 predictions_a)
    if len(targets_a_attm) == 0:
        targets_a_attm = [1]
        predictions_a_attm = [1]

    targets_a_ctrw, predictions_a_ctrw = from_class_order_find_regression(model_order_ctrw, targets_a,
                                                                          predictions_a)
    if len(targets_a_ctrw) == 0:
        targets_a_ctrw = [1]
        predictions_a_ctrw = [1]

    targets_a_fbm, predictions_a_fbm = from_class_order_find_regression(model_order_fbm, targets_a,
                                                                        predictions_a)
    if len(targets_a_fbm) == 0:
        targets_a_fbm = [1]
        predictions_a_fbm = [1]

    targets_a_lw, predictions_a_lw = from_class_order_find_regression(model_order_lw, targets_a,
                                                                      predictions_a)
    if len(targets_a_lw) == 0:
        targets_a_lw = [1]
        predictions_a_lw = [1]

    targets_a_sbm, predictions_a_sbm = from_class_order_find_regression(model_order_sbm, targets_a,
                                                                        predictions_a)
    if len(targets_a_sbm) == 0:
        targets_a_sbm = [1]
        predictions_a_sbm = [1]

    if len(targets_a_sbm)==0 and len(targets_a_lw) == 0 and len(targets_a_fbm) == 0 and len(targets_a_ctrw) == 0 and len(targets_a_attm) == 0:
        x = [1]
        y = [total_mae]
        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["OOD"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=total_mae, color="slategrey")

        mae_plot = {'x': x, 'y': y, 'total_mae': total_mae, 'tick_label': ["OOD"]}

    else:
        attm_rmse = np.sqrt(mean_squared_error(targets_a_attm, predictions_a_attm))
        attm_mae = mean_absolute_error(targets_a_attm, predictions_a_attm)

        ctrw_rmse = np.sqrt(mean_squared_error(targets_a_ctrw, predictions_a_ctrw))
        ctrw_mae = mean_absolute_error(targets_a_ctrw, predictions_a_ctrw)

        fbm_rmse = np.sqrt(mean_squared_error(targets_a_fbm, predictions_a_fbm))
        fbm_mae = mean_absolute_error(targets_a_fbm, predictions_a_fbm)

        lw_rmse = np.sqrt(mean_squared_error(targets_a_lw, predictions_a_lw))
        lw_mae = mean_absolute_error(targets_a_lw, predictions_a_lw)

        sbm_rmse = np.sqrt(mean_squared_error(targets_a_sbm, predictions_a_sbm))
        sbm_mae = mean_absolute_error(targets_a_sbm, predictions_a_sbm)

        x = [1, 2, 3, 4, 5]
        y = [attm_mae, ctrw_mae, fbm_mae, lw_mae, sbm_mae]
        c = total_mae


        ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
        ax2.bar(x, y, color="cornflowerblue", hatch="/", tick_label=["ATTM", "CTRW", "FBM", "LW", "SBM"])
        plt.setp(ax2.get_xticklabels(), fontsize=16)
        plt.setp(ax2.get_yticklabels(), fontsize=16)
        ax2.set_xlabel("model", fontsize=16)
        ax2.set_ylabel("mae", fontsize=16)
        ax2.set_title("Regression", fontsize=16)
        ax2.axhline(y=c, color="slategrey")

        mae_plot = {'x': x, 'y': y, 'total_mae': c,'tick_label':["ATTM", "CTRW", "FBM", "LW", "SBM"]}






    aa = pd.DataFrame(index=range(3), columns=['a', 'b', 'c'])
    step = 0.1
    alen = np.arange(0, 2.00, step)
    F = len(alen)
    data_set1 = np.random.rand(F, F)
    # FF = len(targets_m)
    FF = len(targets_a)
    simu = np.random.randint(100, size=(F, F))
    for z in range(FF):
        if predictions_a[z] > 2:
            predictions_a[z] = 2
        if predictions_a[z] < 0:
            predictions_a[z] = 0.0001

    for j in range(F):
        for i in range(F):
            simu[i][j] = 0
            a_target_low = j * step
            a_target_high = (j + 1) * step
            a_pred_low = i * step
            a_pred_high = (i + 1) * step
            for z in range(FF):
                if ((targets_a[z] > a_target_low) and (targets_a[z] <= a_target_high)):
                    if ((predictions_a[z] > a_pred_low) and (predictions_a[z] <= a_pred_high)):
                        simu[i][j] = simu[i][j] + 1
                    # elif (predictions_a[z] >2):

    ttt = 0
    for j in range(F):
        for i in range(F):
            ttt = ttt + simu[i][j]
    #print(ttt)

    for i in range(F):
        ss = sum(simu[:, i])
        for j in range(F):
            if ss==0:
                data_set1[j, i] =0
            else:
                data_set1[j, i] = float(simu[j, i]) / float(ss)

        # simu[:,i]=simu[:,i]/ss

    # data_set1 = simu

    index1 = []
    for i in range(F):
        if alen[i] == 0.5:
            index1.append('0.5')
        elif alen[i] == 1:
            index1.append('1.0')
        elif alen[i] == 1.5:
            index1.append('1.5')
        else:
            index1.append('')
    column1 = index1

    column1 = pd.CategoricalIndex(index1, ordered=False, categories=["", "0.5", "1.0", "1.5"])
    index1 = column1
    aa = pd.DataFrame(index=index1, columns=column1)

    data_set1 = np.array(data_set1, dtype=np.float)

    for i in range(F):
        aa.iloc[i] = data_set1[i]
        # aa.iloc[i]=aa.iloc[i].astype('float')

    aa = aa.apply(pd.to_numeric, errors='ignore')

    sns.set_context({"figure.figsize": (8, 8)})

    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)

    #axx = sns.heatmap(data=aa, square=True, cmap='coolwarm', vmin=-0.6, vmax=0.4, cbar=False, ax=ax3)
    axx = sns.heatmap(data=aa, square=True, cmap='Blues', cbar=False, ax=ax3)

    axx.invert_yaxis()
    plt.setp(axx.get_xticklabels(), fontsize=16)
    plt.setp(axx.get_yticklabels(), fontsize=16)
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='x', labelsize=20)
    # axx.set_xticks(fontsize=20)
    # axx.set_yticks(fontsize=20)
    axx.set_xlabel('groundtruth', fontsize=16)
    axx.set_ylabel('prediction', fontsize=16)
    axx.set_title("Regression Heat Map", fontsize=16)
    # plt.show()
    heat_a_plot=aa

    from sklearn.metrics import f1_score

    #f1 = f1_score(targets_m, predictions_m, average='micro')
    #print(f1)

    plt.show()

    #
    ########################################################################################################
    load = addre + 'checkpoints/model_best.pth'
    if (os.path.isfile(load_reg)):
        # config['load_model'] = 'output/_2022-12-30_21-35-30_g4L/checkpoints/model_best.pth'
        config['change_output'] = False
        # Initialize optimizer
        # config['l2_reg']
        if config['global_reg']:
            weight_decay = config['l2_reg']
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config['l2_reg']
        # config['optimizer']
        optim_class = get_optimizer(config['optimizer'])
        optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
        model, optimizer, start_epoch = utils.load_model(model, load, optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
        model = model.to(device)


    return test_loader,f1,total_mae,metrics_dict,class_plot,mae_plot,heat_a_plot
