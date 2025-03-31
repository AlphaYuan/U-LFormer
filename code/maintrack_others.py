

import gc
import numpy as np
#import andi
import csv
#ANDI = andi.andi_datasets()
import ruptures as rpt  # 导入ruptures
import torch
from utils1 import prepare_dataset



# Project modules
from options import Options


# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options

import logging
import sys


import os
import traceback
import json
from datetime import datetime
import string
import random
from utils import utils, analysis
from collections import OrderedDict
import time
import pickle
from functools import partial

import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

# Project modules
from options import Options

from utils import utils

from datasets.dataset import collate_superv_for_cp
from running1 import setup, pipeline_factory, validate, check_progress, NEG_METRICS       #running with logitnorm loss
#from running1 import setup, pipeline_factory, validate, check_progress, NEG_METRICS    #running1 without logitnorm loss
from datasets.data import data_factory


from datasets.dataset import ClassiregressionDataset, collate_superv
# from transmodel2 import model_factory
from transmodel_fusion import model_factory
#from transmodel11 import model_factory
from losscompute import get_loss_module
from optimizers import get_optimizer
from torch import optim
from torch.nn import functional as F

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}

#把args中的配置变成字典
def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


class NoFussCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary

    gpus = args.gpus #[0,1,2,3]
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = False
    # Set this to True for WaveNet
    torch.backends.cudnn.benchmark = True 
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    # 回归
    config['model'] = 'ModelFusion'
    config['data_difference'] = False
    config['task'] ='classification'
    config['data_class'] = 'Trackdata'
    config['pattern'] = 'TRAIN'
    config['val_pattern'] = 'TEST'
    config['records_file'] = 'Classification_records.xls'
    config['optimizer'] = 'RAdam'
    config['lr'] = 0.005
    config['epochs'] = 1000
    config['pos_encoding'] = 'learnable'
    config['key_metric'] = 'loss'
    config['whether_pos_encoding'] = True
    config['batch_size'] = 2000 * len(gpus) #500#   #7000 is applied in the paper
    config['d_model']=32
    config['max_seq_len'] = 200
    config['num_layers'] = 6
    config['val_interval'] = 1
    config['num_heads'] = 8
    config['dim_feedforward'] = 256

    config['pred_state'] = False
    config['only_state'] = False
    
    # Dataset Path
    # Trajactory of Length 200
    # config['train_address'] = '/data1/jiangy/andi_tcu/code/andi_2/datasets/andi_set/1104_multi_state/train/train0_Nall_300000_T200_200_K-3_3.csv'
    # config['valid_address'] = '/data1/jiangy/andi_tcu/code/andi_2/datasets/andi_set/1104_multi_state/test/test0_Nall_300000_T200_200_K-3_3.csv'
    # Trajactory of Length 20-200
    config['train_address'] = '/data1/jiangy/andi_tcu/code/andi_2/datasets/andi_set/1110_multi_state/train/merge.csv'
    config['valid_address'] = '/data1/jiangy/andi_tcu/code/andi_2/datasets/andi_set/1110_multi_state/test/merge.csv'

    # input trajectory dimension (x, y; dim=2)
    config['dimension']= 2#*5
    config['tlen'] = 200  # length
    config['tnum'] = 300000  # samples of each model
    config['normalization_layer'] = 'LayerNorm'
    config['cnn_embedding'] = True
    config['cnn_layer'] = 3    #no use in unet
    config['cnn_kernel'] = 3    #no use in unet
    config['cnn_outchannels'] = [20, 32,64]
    trainfilename = config['train_address']
    validfilename = config['valid_address']
    config['cnnencoderkernel']=3
    config['cnnencoderhidden']=32 # hidden dim 
    config['cnnencoderdilation']=5 # dilation depth for wavenet

    # Partition the traning dataset. Change to the next dataset part every 20 epoch
    # For bigger dataset (not in our paper)
    config['train_folder'] = '/data1/jiangy/andi_data/0529/train/'
    config['test_folder'] = '/data1/jiangy/andi_data/0529/test/'
    config['part_data'] = False
    config['part_num'] = 5
    config['change_epoch'] = 20

    # Custom loss weight
    config['weight'] = True # True
    config['add_noise'] = False # 加上N(0,1)的噪声
    config['loss'] = 'L1'
    config['pred_cp'] = False
    # config['log'] = 'ln'
    #config['normalization_layer'] = 'LayerNorm'
    # config['load_model'] = '/data1/jiangy/andi_tcu/code/andi_2/challenge_output/_2024-07-09_10-53-45_7tB/checkpoints/model_last_seg.pth'
    # config['resume'] = True
    # config['lr'] = 0.00015625
    if config['model'] == 'ModelFusion_deep':
        config['d_model'] = int(config['d_model'] * 1.5)
        config['cnnencoderhidden'] = int(config['cnnencoderhidden'] * 1.5)





    print(config)

    total_epoch_time = 0
    total_eval_time = 0
    outchannel = config['cnn_outchannels'][:]

    tlen = config['tlen']  #
    tnum = config['tnum'] #

    total_start_time = time.time()
    logger.info("Loading and preprocessing data ...")


    with open(trainfilename, 'r') as fp:  #
        data = list(csv.reader(fp))  #
        train_set1 = []
        for i in range(len(data)):
            t = []
            for j in range(len(data[i])):
                t.append(float(data[i][j]))
            train_set1.append(t)

    with open(validfilename, 'r') as fp:
        data = list(csv.reader(fp))  #
        valid_set1 = []
        for i in range(len(data)):
            t = []
            for j in range(len(data[i])):
                t.append(float(data[i][j]))
            valid_set1.append(t)


    output_dir=config['output_dir']

    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)


    train_set1 = np.asarray(train_set1, dtype=object)
    valid_set1 = np.asarray(valid_set1, dtype=object)
    dim=config['dimension']
    dataset_class, collate_fn, runner_class , collate_fn_for_weight= pipeline_factory(config)
    ########################################################################################################################
    # Create model
    logger.info("Creating model ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(valid_set1, dim=dim, n_proc=config['n_proc'],
                         limit_size=config['limit_size'], config=config)  # (sequencelenth*sample_num,feat_dim)

    model = model_factory(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
    logger.info('number of params (M): %.2f' % (utils.count_parameters(model) / 1.e6))

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])

    #optimizer = RAdam(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    if config['key_metric']=='loss':
        mode='min'
    if config['key_metric']=='accuracy':
        mode = 'max'


    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=True, threshold=1e-4, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-20, eps=1e-15)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`  默认值 config['lr_step']=[10000], config['lr_fractor']=[0.1]  也就是当epoch=10000时候，lr=0.1*lr
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if config['load_model'] is not None:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.to(device)


    loss_module = get_loss_module(config)


    ########################################################################################################################
    # Create dataset
    def track_data(config,filename):
        with open(filename, 'r') as fp:
            data = list(csv.reader(fp))
            train_set1 = []
            for i in range(len(data)):
                t = []
                for j in range(len(data[i])):
                    t.append(float(data[i][j]))
                train_set1.append(t)
        if len(train_set1)<500000:
            train_set1 = np.asarray(train_set1,  dtype=object)
        else:
            train_set1 = random.sample(train_set1, 300000)
            train_set1 = np.asarray(train_set1,  dtype=object)


        data_class = data_factory['Trackdata']
        my_data = data_class(train_set1, dim=config['dimension'], n_proc=-1,
                             limit_size=None, config=config)  # (sequencelenth*sample_num,feat_dim)
        train_indices = my_data.all_IDs

        # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
        # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0

        logger.info("{} samples will be used for testing".format(len(train_indices)))

        train_dataset = dataset_class(my_data, train_indices)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_fn(x, max_len=model.module.max_len))
        if config['pred_cp']:
            train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_superv_for_cp(x, max_len=model.module.max_len))
        if not config['weight']:
            return train_dataset,train_loader,train_indices, None

        train_loader_for_weight = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_fn_for_weight(x, max_len=model.module.max_len))
        train_weight_mask=[]
        for i, batch in enumerate(train_loader_for_weight):
            _, _, _, _, weight_mask = batch
            weight_mask = weight_mask.to(device)
            train_weight_mask.append(weight_mask)
        train_weight_mask = torch.cat(train_weight_mask, dim=0)
        # JY
        # return mask
        return train_dataset,train_loader,train_indices,train_weight_mask
        # return train_loader_for_weight, train_loader, train_indices

    # train_dataset, train_loader,train_indices =track_data(config, trainfilename)
    # val_dataset, val_loader,val_indices  = track_data(config, validfilename)
    train_dataset, train_loader,train_indices,t_mask =track_data(config, trainfilename)

    val_dataset, val_loader,val_indices,v_mask  = track_data(config, validfilename)

    # train_weight_mask=[]
    # for i, batch in enumerate(train_loader_for_weight):
    #     X, targets, padding_masks, IDs, weight_mask = batch
    #     weight_mask = weight_mask.to(device)
    #     train_weight_mask.append(weight_mask)
    # t_mask = torch.cat(train_weight_mask, dim=0)

    # val_weight_mask = []
    # for i, batch in enumerate(val_loader_for_weight):
    #     X, targets, padding_masks, IDs, weight_mask = batch
    #     weight_mask = weight_mask.to(device)
    #     val_weight_mask.append(weight_mask)
    # v_mask = torch.cat(val_weight_mask, dim=0)




    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))


    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices)}, f, indent=4)


    ########################################################################################################################




    #########################################################################################################################

    #########################################################################################################################

    trainer = runner_class(model, train_loader, device, loss_module, optimizer,weight_mask=t_mask,l2_reg=output_reg,
                           print_interval=config['print_interval'], console=config['console'],model_data=my_data,config=config)
    val_evaluator = runner_class(model, val_loader, device, loss_module, weight_mask=v_mask,
                                 print_interval=config['print_interval'], console=config['console'],model_data=my_data,config=config)

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])   #
    #
    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)


    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    # Starting Training
    logger.info('Starting training...')
    multi_loss_list = []
    metric_list1=[]
    metric_list2 = []
    metric_list_train = []
    metric_loss_valid = []

    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        if config['part_data'] and epoch != start_epoch+1 and epoch % config['change_epoch'] == 0:
            print("change dataloader")
            data_idx = int(epoch / config['change_epoch']) % config['part_num']
            train_file = os.path.join(config['train_folder'], 'merge_sample_part{}.csv'.format(data_idx))
            del train_dataset, train_loader, train_indices
            gc.collect()
            trainer.change_dataloader(None, None) # change val_loader, reduce memory use
            train_dataset, train_loader,train_indices,t_mask =track_data(config, train_file)
            trainer.change_dataloader(train_loader, t_mask)
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info(
            "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))


        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))  #

        if epoch % 20 == 0:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}_seg.pth'.format(epoch)), epoch, model, optimizer)

        utils.save_model(os.path.join(config['save_dir'], 'model_{}_seg.pth'.format(mark)), epoch, model, optimizer)
        # Learning rate scheduling
        reduce_schedule.step(aggr_metrics_train['loss'])
        multi_loss_list.append(optimizer.param_groups[0]["lr"])
        logger.info("learning rate: {} ".format(optimizer.param_groups[0]["lr"]))

        #reduce_schedule.step(aggr_metrics_train['loss'])
        #multi_loss_list.append(optimizer.param_groups[0]["lr"])
        if config['key_metric'] == 'accuracy':
            #reduce_schedule.step(aggr_metrics_train[config['key_metric']].item())
            #multi_loss_list.append(optimizer.param_groups[0]["lr"])
            metric_list1.append(aggr_metrics_val[config['key_metric']].item())
            metric_list2.append(aggr_metrics_train[config['key_metric']].item())
        else:
            #reduce_schedule.step(aggr_metrics_train[config['key_metric']])
            #multi_loss_list.append(optimizer.param_groups[0]["lr"])
            metric_list1.append(aggr_metrics_val[config['key_metric']])
            metric_list2.append(aggr_metrics_train[config['key_metric']])



        metric_list_train.append(aggr_metrics_train['loss'])
        metric_loss_valid.append(aggr_metrics_val['loss'])



        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

        with open(os.path.join(config['output_dir'], (config[
                                                          'task'] + 'len:{0}_num:{1}_dmodel:{2}_nlayer:{3}_nhead:{4}_epoch{5}_all_metrics_log.csv'.format(
                tlen, tnum, config['d_model'], config['num_layers'], config['num_heads'], config['epochs']))), 'w',
                  encoding='utf-8', newline='') as f:
            if config['key_metric'] == 'loss':
                writer = csv.writer(f)
                writer.writerow(['val_loss', 'train_loss', 'reduce_lr'])
                ttt = zip(metric_list1, metric_list2, multi_loss_list)
                writer.writerows(ttt)
            if config['key_metric'] == 'accuracy':
                writer = csv.writer(f)
                writer.writerow(['val_accuracy', 'train_accuracy', 'reduce_lr', 'val_loss', 'train_loss'])
                ttt = zip(metric_list1, metric_list2, multi_loss_list, metric_loss_valid, metric_list_train)
                writer.writerows(ttt)

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    import matplotlib.pyplot as plt
    plt.subplot(151)
    if config['key_metric'] == 'loss':
        plt.plot(range(len(metric_list1)), metric_list1, label="val_loss")
        plt.legend()
    if config['key_metric'] == 'accuracy':
        plt.plot(range(len(metric_list1)), metric_list1, label='val_accuracy')
        plt.legend()

    plt.subplot(152)
    if config['key_metric'] == 'loss':
        plt.plot(range(len(metric_list2)), metric_list2, label="train_loss")
        plt.legend()
    if config['key_metric'] == 'accuracy':
        plt.plot(range(len(metric_list2)), metric_list2, label='train_accuracy')
        plt.legend()

    plt.subplot(153)
    plt.plot(range(len(multi_loss_list)), multi_loss_list, label="reduce_lr")
    plt.legend()

    plt.subplot(154)
    plt.plot(range(len(metric_list_train)), metric_list_train, label='train_loss')
    plt.legend()

    plt.subplot(155)
    plt.plot(range(len(metric_loss_valid)), metric_loss_valid, label='val_loss')
    plt.legend()

    plt.show()


    with open(os.path.join(config['output_dir'], (config['task']+'logitnorm'+'len:{0}_num:{1}_dmodel:{2}_nlayer:{3}_nhead:{4}_epoch{5}_all_metrics_log.csv'.format(tlen,tnum,config['d_model'],config['num_layers'],config['num_heads'],config['epochs']))), 'w',
              encoding='utf-8', newline='') as f:
        if config['key_metric'] == 'loss':

            writer = csv.writer(f)
            writer.writerow(['val_loss', 'train_loss', 'reduce_lr'])
            ttt = zip(metric_list1, metric_list2,multi_loss_list)
            writer.writerows(ttt)
        if config['key_metric'] == 'accuracy':

            writer = csv.writer(f)
            writer.writerow(['val_accuracy', 'train_accuracy', 'reduce_lr','val_loss', 'train_loss'])
            ttt = zip(metric_list1, metric_list2,multi_loss_list,metric_loss_valid,metric_list_train)
            writer.writerows(ttt)
    a=0


a=3
