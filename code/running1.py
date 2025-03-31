'''
no logitnorm loss
'''

from losscompute import MaskedMSELoss
import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn
from torch.nn import functional as F
from losscompute import get_loss_module

from utils import utils, analysis
from losscompute import l2_reg_loss
from datasets.dataset import ClassiregressionDataset, collate_unsuperv, \
    collate_superv, ClassiregressionDataset_no_data_difference
from datasets.dataset import ClassiregressionDataset, collate_superv,collate_superv_for_weight,collate_superv_for_cp

from andi_datasets.utils_challenge import label_continuous_to_list

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config['task']

    if 'data_difference' in config and config['data_difference'] == False:
        return ClassiregressionDataset_no_data_difference, collate_superv, SupervisedRunner, collate_superv_for_weight
    if (task == "classification") or (task == "regression"):
        return ClassiregressionDataset, collate_superv, SupervisedRunner, collate_superv_for_weight
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


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


def fold_evaluate(dataset, model, device, loss_module, target_feats, config, dataset_name):
    allfolds = {'target_feats': target_feats,
                # list of len(num_folds), each element: list of target feature integer indices
                'predictions': [],
                # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) prediction per sample
                'targets': [],
                # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) target/original input per sample
                'target_masks': [],
                # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) boolean mask per sample
                'metrics': [],  # list of len(num_folds), each element: (num_samples, num_metrics) metric per sample
                'IDs': []}  # list of len(num_folds), each element: (num_samples,) ID per sample

    for i, tgt_feats in enumerate(target_feats):

        dataset.mask_feats = tgt_feats  # set the transduction target features

        loader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_unsuperv(x, max_len=config['max_seq_len']))

        evaluator = UnsupervisedRunner(model, loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

        logger.info("Evaluating {} set, fold: {}, target features: {}".format(dataset_name, i, tgt_feats))
        aggr_metrics, per_batch = evaluate(evaluator)

        metrics_array = convert_metrics_per_batch_to_per_sample(per_batch['metrics'], per_batch['target_masks'])
        metrics_array = np.concatenate(metrics_array, axis=0)
        allfolds['metrics'].append(metrics_array)
        allfolds['predictions'].append(np.concatenate(per_batch['predictions'], axis=0))
        allfolds['targets'].append(np.concatenate(per_batch['targets'], axis=0))
        allfolds['target_masks'].append(np.concatenate(per_batch['target_masks'], axis=0))
        allfolds['IDs'].append(np.concatenate(per_batch['IDs'], axis=0))

        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        for m, metric_name in enumerate(list(aggr_metrics.items())[1:]):
            logger.info("{}:: Mean: {:.3f}, std: {:.3f}".format(metric_name, metrics_mean[m], metrics_std[m]))

    pred_filepath = os.path.join(config['pred_dir'], dataset_name + '_fold_transduction_predictions.pickle')
    logger.info("Serializing predictions into {} ... ".format(pred_filepath))
    with open(pred_filepath, 'wb') as f:
        pickle.dump(allfolds, f, pickle.HIGHEST_PROTOCOL)


def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
    """
    Args:
        metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,) metric per element
        target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active, 0s ignore
    Returns:
        metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
    """
    metrics_array = []
    for b, batch_target_masks in enumerate(target_masks):
        num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
        batch_metrics = np.stack(metrics[b], axis=1)  # (num_active_in_batch, num_metrics)
        ind = 0
        metrics_per_sample = np.zeros((len(num_active_per_sample), batch_metrics.shape[1]))  # (batch_size, num_metrics)
        for n, num_active in enumerate(num_active_per_sample):
            new_ind = ind + num_active
            metrics_per_sample[n, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
            ind = new_ind
        metrics_array.append(metrics_per_sample)
    return metrics_array


def evaluate(evaluator):
    """Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    print()
    print_str = 'Evaluation Summary: '
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    return aggr_metrics, per_batch


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    global val_times  # global var
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Epoch {} Validation Summary: '.format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)

    if condition:
        best_value = aggr_metrics[config['key_metric']]

        if config['task']=='classification':
            utils.save_model(os.path.join(config['save_dir'], 'model_best_seg.pth'), epoch, val_evaluator.model)
        else:
            utils.save_model(os.path.join(config['save_dir'], 'model_best_reg.pth'), epoch, val_evaluator.model)

        #utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


def check_progress(epoch):
    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, weight_mask=None,l2_reg=None, print_interval=10,
                 console=True, model_data=None,config=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.weight_mask=weight_mask

        self.epoch_metrics = OrderedDict()

        self.mydata = model_data

        self.config=config

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets,
                                    target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # TODO: for debugging
            # input_ok = utils.check_tensor(X, verbose=False, zero_thresh=1e-8, inf_thresh=1e4)
            # if not input_ok:
            #     print("Input problem!")
            #     ipdb.set_trace()
            #
            # utils.check_model(self.model, verbose=False, stop_on_error=True)

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets,
                                    target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch
            if keep_all:
                per_batch['target_masks'].append(target_masks.cpu().numpy())
                per_batch['targets'].append(targets.cpu().numpy())
                per_batch['predictions'].append(predictions.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])
                per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):

    def __init__(self, *args, **kwargs):

        super(SupervisedRunner, self).__init__(*args, **kwargs)

        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        else:
            self.classification = False
    # training with dataset part
    def change_dataloader(self, dataloader, mask):
        self.dataloader = dataloader
        self.weight_mask = mask
    
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()
        if self.mydata != None:
            self.num_labels = len(self.mydata.class_names)

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):

            '''X, targets, padding_masks, IDs, weight_mask = batch
            targets = targets.to(self.device)
            weight_mask = weight_mask.to(self.device)'''

            X, targets, padding_masks, IDs = batch
            targets = targets.to(self.device)
            #padding_masks = padding_masks.to(self.device)  # 0s: ignore
            # regression: (batch_size, num_labels);
            predictions, _, _ = self.model(X.to(self.device), padding_masks)
            loss_module_m = NoFussCrossEntropyLoss(reduction='none')

            if self.config['pred_cp']:
                loss_ce = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 9])).to(self.device)
                ce_cp = loss_ce(predictions.permute(0, 2, 1), targets[:, :, 3].long())
                loss = ce_cp
                mean_loss = loss
                batch_loss = loss
            elif self.weight_mask is None:
                lossmae = get_criterion('MAE')
                lossmse = get_criterion('MSE')
                lossce = torch.nn.CrossEntropyLoss(weight=torch.Tensor([9, 9, 1, 9])).to(self.device) #get_criterion('CE')
                loss_mse_weight = get_criterion('weighted_MSE')
                if self.config['only_state']:
                    ce_state_ce = lossce(predictions.permute(0,2,1), targets[:, :, 2].long())
                    loss=ce_state_ce
                else:
                    mae_a_mae = lossmae(predictions[:, :, 0], targets[:, :, 0]) # mse
                    mse_k_mse = lossmse(predictions[:, :, 1], targets[:, :, 1])
                    loss=mae_a_mae +mse_k_mse
                    # mse_a_mse, mse_k_mse = loss_mse_weight(predictions, targets)
                    # loss = mse_a_mse + mse_k_mse
                    if self.config['pred_state']:
                        ce_state_ce = lossce(predictions[:, :, 2:].permute(0,2,1), targets[:, :, 2].long())
                        loss+=ce_state_ce
                mean_loss=loss
                batch_loss=loss
            else:
                #weight_mask1 = self.weight_mask[i]
                weight_mask1=self.weight_mask[IDs]
                pred_a = predictions[:, :, 0]
                targ_a = targets[:, :, 0]
                pred_k = predictions[:, :, 1]
                targ_k = targets[:, :, 1]

                weight_a = weight_mask1[:, :, 0]
                weight_k = weight_mask1[:, :, 1]

                loss_module_a = torch.nn.L1Loss(reduction='none')
                loss_module_k = torch.nn.MSELoss(reduction='none')

                lossa = loss_module_a(pred_a, targ_a)
                lossk = loss_module_k(pred_k, targ_k)

                wlossa = torch.mul(lossa, weight_a)
                wlossk = torch.mul(lossk, weight_k)

                non_zero_indices = weight_a.nonzero()
                # counting the num of non-zero
                num_non_zero_a = non_zero_indices.size(0)

                mae_a_mean = torch.sum(wlossa) / num_non_zero_a
                mse_k_mean = torch.sum(wlossk) / num_non_zero_a

                loss = mae_a_mean + mse_k_mean
                mean_loss = loss
                batch_loss = loss
                

            if self.l2_reg:
                    total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.


            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            # metrics = {"loss": mean_loss.item()}
            metrics = {"loss": mean_loss.item()}#, 'a': mse_a_mse.cpu().item(), 'k': mse_k_mse.cpu().item(), 'st': ce_state_ce.cpu().item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                #total_samples += len(loss)
                # total_samples += len(loss_a)
                total_samples += 1
                epoch_loss += batch_loss.item()  # add total loss of batch

        #########################################################################################################
        per_batch1 = {'target_masks': [], 'targets': [],  'predictions': [],
                      'metrics': [], 'IDs': []}

        per_batch1['targets'].append(targets.cpu().numpy())
        per_batch1['predictions'].append(predictions.detach().cpu().numpy())
        per_batch1['metrics'].append([loss.detach().cpu().numpy()])
        # per_batch1['metrics'].append([loss_a.detach().cpu().numpy()])
        per_batch1['IDs'].append(IDs)


        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'target_masks': [], 'targets': [],  'predictions': [],
                     'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):

            '''X, targets, padding_masks, IDs,weight_mask = batch
            targets = targets.to(self.device)
            weight_mask = weight_mask.to(self.device)'''

            X, targets, padding_masks, IDs = batch
            targets = targets.to(self.device)
            # padding_masks = padding_masks.to(self.device)  # 0s: ignore
            # regression: (batch_size, num_labels);
            predictions, _, _ = self.model(X.to(self.device), padding_masks)
            loss_module_m = NoFussCrossEntropyLoss(reduction='none')
            # loss_module_m = LogitNormLoss(self.device,0.4)
            loss_module_a = torch.nn.MSELoss(reduction='none')
            
            if self.config['pred_cp']:
                loss_ce = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 9])).to(self.device)
                ce_cp = loss_ce(predictions.permute(0, 2, 1), targets[:, :, 3].long())
                loss = ce_cp
                mean_loss = loss.cpu().item()
                batch_loss = loss.cpu().item()
            elif self.weight_mask is None:
                lossmae = get_criterion('MAE')
                lossmse = get_criterion('MSE')
                # lossce = get_criterion('CE')
                lossce = torch.nn.CrossEntropyLoss(weight=torch.Tensor([9, 9, 1, 9])).to(self.device) #get_criterion('CE')
                loss_mse_weight = get_criterion('weighted_MSE')
                if self.config['only_state']:
                    mse_state_mse = lossce(predictions.permute(0,2,1), targets[:, :, 2].long())
                    loss = mse_state_mse
                else:
                    mse_a_mse = lossmse(predictions[:, :, 0], targets[:, :, 0])
                    mse_k_mse = lossmse(predictions[:, :, 1], targets[:, :, 1])
                    loss = mse_a_mse  + mse_k_mse
                    # mse_a_mse, mse_k_mse = loss_mse_weight(predictions, targets)
                    # loss = mse_a_mse + mse_k_mse
                    if self.config['pred_state']:
                        mse_state_mse = lossce(predictions[:, :, 2:].permute(0,2,1), targets[:, :, 2].long())
                        loss += mse_state_mse
                mean_loss = loss.cpu().item()
                batch_loss = loss.cpu().item()
            else:
                #weight_mask1=self.weight_mask[i]
                weight_mask1 = self.weight_mask[IDs]
                pred_a = predictions[:, :, 0]
                targ_a = targets[:, :, 0]
                pred_k = predictions[:, :, 1]
                targ_k = targets[:, :, 1]

                weight_a = weight_mask1[:, :, 0]
                weight_k = weight_mask1[:, :, 1]

                loss_module_a = torch.nn.L1Loss(reduction='none')
                loss_module_k = torch.nn.MSELoss(reduction='none')

                lossa = loss_module_a(pred_a, targ_a)
                lossk = loss_module_k(pred_k, targ_k)

                wlossa = torch.mul(lossa, weight_a)
                wlossk = torch.mul(lossk, weight_k)

                non_zero_indices = weight_a.nonzero()

                num_non_zero_a = non_zero_indices.size(0)

                mae_a_mean = torch.sum(wlossa) / num_non_zero_a
                mse_k_mean = torch.sum(wlossk) / num_non_zero_a

                loss = mae_a_mean + mse_k_mean
                mean_loss = loss
                batch_loss = loss

                mean_loss = loss.cpu().item()
                batch_loss = loss.cpu().item()

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            #batch_loss = torch.sum(loss).cpu().item()
            #mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            '''loss = loss_a * 5

            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)'''

            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(predictions.cpu().numpy())
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}#, 'a': mse_a_mse.cpu().item(), 'k': mse_k_mse.cpu().item(), 'st': mse_state_mse.cpu().item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            #total_samples += len(loss)
            total_samples += 1
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics

class NoFussCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class LogitNormLoss(torch.nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target.long().squeeze())

class RMSLELoss(nn.Module):
    def __init__(self, loss='MSE'):
        super().__init__()
        self.loss = nn.MSELoss()
        if loss == 'MAE':
            self.loss = nn.L1Loss()

    def forward(self, pred, actual):
        # return torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(actual + 1)))
        # if torch.min(pred) < 0:
        #     return self.loss(torch.log(torch.clip(pred, 0) + 1), torch.log(actual + 1))
        return self.loss(torch.log(pred + 1), torch.log(actual + 1))
    
class weighted_Loss(nn.Module):
    def __init__(self, loss='MSE'):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        if loss == 'MAE':
            self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, pred, actual):
        weights = torch.ones((actual.shape[0], actual.shape[1]), device=pred.device)
        for i in range(actual.shape[0]):
            # changepoints, _, _, _ = label_continuous_to_list(actual[i])
            labs = actual[i, :, :].cpu().numpy()
            CP = np.argwhere((labs[:-1, :] != labs[1:, :]).sum(1) != 0).flatten()+1
            weights[i][CP] = 2
        # loss_a = self.loss_fn(pred[:, :, 0], actual[:, :, 0])
        # loss_k = self.loss_fn(pred[:, :, 1], actual[:, :, 1])
        # print(pred.shape, actual.shape, weights.shape)
        loss_a_weighted = torch.mean(((pred[:, :, 0] - actual[:, :, 0]) ** 2) * weights)
        loss_k_weighted = torch.mean(((pred[:, :, 1] - actual[:, :, 1]) ** 2) * weights)
        # loss = loss_a_weighted + loss_k_weighted
        return loss_a_weighted, loss_k_weighted



def get_criterion(loss='MAE'):
    if loss == 'MAE':
        return nn.L1Loss()
    elif loss == 'MSE':
        return nn.MSELoss()
    elif loss == 'MSLE':
        return RMSLELoss()
    elif loss == 'MALE':
        return RMSLELoss('MAE')
    elif loss == 'CE':
        return nn.CrossEntropyLoss(ignore_index=5)
    elif loss == 'weighted_MSE':
        return weighted_Loss('MSE')
    elif loss == 'weighted_MAE':
        return weighted_Loss('MAE')