# shortest path distance prediction - regression

import os
import math
import logging
import random
import time
import pickle
from itertools import combinations, permutations
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.utils import dense_to_sparse

from config import Config as Config
from utils import tool_funcs
from utils.osm_loader import OSMLoader
from task.base_task import BaseTask
from task.spd_dataprocess import calculate_spd_dict


class DistRegression(nn.Module):
    def __init__(self, n_dim):
        super(DistRegression, self).__init__()
        self.enc = nn.Sequential(nn.Linear(n_dim, n_dim), 
                                nn.ReLU())
        self.outlayers = nn.Sequential(nn.Linear(n_dim, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 1),
                                    nn.ReLU())

    def forward(self, inputs):
        inputs = self.enc(inputs)
        n_pairs = int(inputs.shape[0] / 2)
        return self.outlayers( inputs[n_pairs:] - inputs[:n_pairs] ).squeeze(dim = 1)


# Following RNE paper, use sampled pairs.
# not L1-loss, supporting directed graph, 
class SPD(BaseTask):
    def __init__(self, osm_data, embs, encoder):
        super(SPD, self).__init__()

        self.distregression = None

        # dont use model's osm_data, 
        # it may contain extra edges between segments, 
        # e.g. spatial_connections used in SARN.
        self.osm_data = osm_data
        self.seg_ids = self.osm_data.segid_in_adj_segments_graph # list
        self.seg_id_to_idx = self.osm_data.seg_id_to_idx_in_adj_seg_graph
        self.spd_pairs_dataset = spd_pairs_dataset()

        self.embs = embs
        self.encoder = encoder
        self.encoder_mode = Config.task_encoder_mode

        self.checkpoint_filepath = '{}/exp/snapshots/{}_spd_{}_best{}.pkl' \
                                    .format(Config.root_dir, Config.dataset_prefix, \
                                            Config.task_encoder_model, Config.dumpfile_uniqueid)


    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_spd start. @={:.0f}, encoder_mode={}".format(training_starttime, self.encoder_mode))

        in_dim = Config.sarn_embedding_dim 
        self.distregression = DistRegression(in_dim)
        self.distregression.to(Config.device)
        self.distregression.train()
        self.criterion = nn.MSELoss()
        self.criterion.to(Config.device)
        l1loss = nn.L1Loss(reduction = 'none') # for metric only
        l1loss.to(Config.device)

        if self.encoder_mode == 'dump':
            optimizer = torch.optim.Adam(self.distregression.parameters(), \
                                        lr = Config.spd_learning_rate, \
                                        weight_decay = Config.spd_learning_weight_decay)
        elif self.encoder_mode == 'finetune':
            if Config.task_encoder_model == 'SARN_ft':
                optimizer = torch.optim.Adam( \
                                        [ {'params': self.distregression.parameters(), \
                                            'lr': Config.spd_learning_rate, \
                                            'weight_decay': Config.spd_learning_weight_decay}, \
                                          {'params': self.encoder.model.encoder_q.layer_out.parameters(), \
                                            'lr': Config.spd_learning_rate * Config.task_finetune_lr_rescale} \
                                        ])
        
        best_epoch = 0
        best_loss_train = 10000000.0
        best_mae_eval = 10000000.0
        best_mre_eval = 10000000.0
        bad_counter = 0
        bad_patience = Config.spd_training_bad_patience

        for i_ep in range(Config.spd_epoch):
            _time_ep = time.time()
            train_losses = []
            train_maes = []
            train_mres = []
            train_gpu = []
            train_ram = []

            self.distregression.train()

            if Config.spd_learning_rated_adjusted:
                if Config.task_encoder_model == 'SARN_ft':
                    _degraded = 0.5 * (1. + math.cos(math.pi * i_ep / Config.spd_epoch))
                    optimizer.param_groups[0]['lr'] = Config.spd_learning_rate * _degraded
                    optimizer.param_groups[1]['lr'] = Config.spd_learning_rate * Config.task_finetune_lr_rescale * _degraded
                else:
                    tool_funcs.adjust_learning_rate(optimizer, Config.spd_learning_rate, i_ep, Config.spd_epoch)


            for i_batch, batch in enumerate(self.spd_dataset_generator_batchi(self.spd_pairs_dataset['trains_segidx'], \
                                                                            self.spd_pairs_dataset['trains_dists'])):
                _time_batch = time.time()
                optimizer.zero_grad()
                task_loss, model_loss = 0.0, 0.0

                sub_seg_idx, pair_dist = batch

                if self.encoder_mode == 'dump':
                    sub_embs = self.embs[sub_seg_idx]
                elif self.encoder_mode == 'finetune':
                    sub_embs = self.encoder.finetune_forward(sub_seg_idx, True)

                outs = self.distregression(sub_embs) # [batch_size * 2, hidden_size]
                pred_l1_dist = outs  # tensor; [batch_size]
                pair_dist = torch.tensor(pair_dist, dtype = torch.float, device = Config.device)

                loss_train = self.criterion(pred_l1_dist, pair_dist)

                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                with torch.no_grad():
                    mae = l1loss(pred_l1_dist, pair_dist)
                    mre = mae / pair_dist

                    train_maes.append( torch.mean(mae).item() )
                    train_mres.append( torch.mean(mre).item() )

                if i_batch % 100 == 0:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.0f}({:.0f}/{:.4f}), "
                                    "mre={:.4f}, mae={:.4f}, @={:.3f}, gpu={}, ram={}" \
                                    .format(i_ep, i_batch, train_losses[-1], task_loss, model_loss, \
                                            train_mres[-1], train_maes[-1], \
                                            time.time()-_time_batch, \
                                            tool_funcs.GPUInfo.mem(), \
                                            tool_funcs.RAMInfo.mem()))

            logging.info("training. i_ep={}, avg_train_loss={:.0f}, mre={:.4f}, mae={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), 
                                tool_funcs.mean(train_mres), tool_funcs.mean(train_maes),
                                time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)
            
            # early stopping
            metrics = self.test()
            mre_ep_eval = metrics['mre']
            mae_ep_eval = metrics['mae']
            logging.info("eval.     i_ep={}, mre={:.4f}, mae={:.4f}".format(i_ep, mre_ep_eval, mae_ep_eval))

            if  mre_ep_eval < best_mre_eval:
                best_epoch = i_ep
                best_mre_eval = mre_ep_eval
                best_mae_eval = mae_ep_eval
                best_loss_train = tool_funcs.mean(train_losses)
                bad_counter = 0
                if self.encoder_mode == 'finetune':
                    if Config.task_encoder_model == 'SARN_ft':
                        torch.save({ "encoder.feat_emb" : self.encoder.feat_emb.state_dict(),
                                    "encoder.encoder_q" : self.encoder.model.encoder_q.state_dict(),
                                    "distregression": self.distregression.state_dict()}, 
                                    self.checkpoint_filepath)
                else:
                    torch.save({'distregression': self.distregression.state_dict()}, 
                                self.checkpoint_filepath)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == Config.spd_epoch:
                training_endtime = time.time()
                logging.info("training end. best_epoch={}, best_mre_eval={:.4f}, "
                            "best_mae_eval={:.4f}, best_loss_train={:.6f}" \
                            .format(best_epoch, best_mre_eval, best_mae_eval, best_loss_train))
                break


        checkpoint = torch.load(self.checkpoint_filepath)
        self.distregression.load_state_dict(checkpoint['distregression'])
        self.distregression.to(Config.device)
        self.distregression.eval()
        if self.encoder_mode == 'finetune':
            if Config.task_encoder_model == 'SARN_ft':
                self.encoder.feat_emb.load_state_dict(checkpoint['encoder.feat_emb'])
                self.encoder.model.encoder_q.load_state_dict(checkpoint['encoder.encoder_q'])

        metrics = self.test()
        logging.info("test. mre={:.4f}, mae={:.4f}, @={:.3f}" \
                    .format(metrics['mre'], metrics['mae'], metrics['task_test_time']))

        metrics.update({'task_train_time': training_endtime - training_starttime, \
                        'task_train_gpu': training_gpu_usage, \
                        'task_train_ram': training_ram_usage})
        return metrics


    @torch.no_grad()
    def test(self):
        test_starttime = time.time()

        self.distregression.eval()

        test_maes, test_mres = [], []
        test_gpus, test_rams = [], []
        l1loss = nn.L1Loss(reduction = 'none') # for metric only
        l1loss.to(Config.device)

        for i_batch, batch in enumerate(self.spd_dataset_generator_batchi(self.spd_pairs_dataset['tests_segidx'], \
                                                                        self.spd_pairs_dataset['tests_dists'])):
            sub_seg_idx, pair_dist = batch

            if self.encoder_mode == 'dump':
                sub_embs = self.embs[sub_seg_idx]
            elif self.encoder_mode == 'finetune':
                sub_embs = self.encoder.finetune_forward(sub_seg_idx, False)

            outs = self.distregression(sub_embs) # [batch_size * 2, hidden_size]
            pred_l1_dist = outs # tensor; [batch_size]
            pair_dist = torch.tensor(pair_dist, dtype = torch.float, device = Config.device)

            mae = l1loss(pred_l1_dist, pair_dist)
            mre = torch.mean(mae / pair_dist)
                
            test_maes.append( torch.mean(mae).item() )
            test_mres.append( torch.mean(mre).item() )
            test_gpus.append(tool_funcs.GPUInfo.mem()[0])
            test_rams.append(tool_funcs.RAMInfo.mem())
        
        test_endtime = time.time()

        return {'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': tool_funcs.mean(test_gpus), \
                'task_test_ram': tool_funcs.mean(test_rams), \
                'mre': tool_funcs.mean(test_mres), 'mae': tool_funcs.mean(test_maes)}


    def spd_dataset_generator_batchi(self, idx_pairs, dists):
        cur_index = 0
        len_dataset = len(idx_pairs)

        while cur_index < len_dataset:
            end_index = cur_index + Config.spd_batch_size \
                                if cur_index + Config.spd_batch_size < len_dataset \
                                else len_dataset

            sub_idx_pairs = idx_pairs[cur_index: end_index]
            a, b = zip(*sub_idx_pairs)
            sub_idx_pairs = list(a) + list(b)
            sub_dists = dists[cur_index: end_index]

            yield sub_idx_pairs, sub_dists
            cur_index = end_index


    @staticmethod
    def get_spd_dict(osm_data):
        _spd_file = '{}/data/{}_spd_distance_dict.pickle'.format(Config.root_dir, Config.dataset_prefix)
        if os.path.exists(_spd_file):
            with open(_spd_file, 'rb') as fh:
                lst_spd = pickle.load(fh)
        else:
            _t = time.time()
            lst_spd = calculate_spd_dict(osm_data)
            logging.info("Compute spd between all seg pairs. #seg={}, shape={}, @={:.3f}"\
                        .format(len(osm_data.segid_in_adj_segments_graph), 
                                (len(lst_spd),len(lst_spd[0])),
                                time.time() - _t) )
            with open(_spd_file, 'wb') as fh:
                pickle.dump(lst_spd, fh, protocol = pickle.HIGHEST_PROTOCOL)
        logging.info("spd data loaded.")
        return np.array(lst_spd, dtype = np.float)


def spd_pairs_dataset():
    spd_pairs_file = '{}/data/{}_spd_pairs.pickle'.format(Config.root_dir, Config.dataset_prefix)
    if os.path.exists(spd_pairs_file):
        with open(spd_pairs_file, 'rb') as fh:
            _time = time.time()
            rtn = pickle.load(fh)
            logging.info('spd_pairs_dataset loaded. @={:.0f}, #={:d}'.format(time.time() - _time, len(rtn['segidx']) ))
    else:
        _time = time.time()
        osm_data = OSMLoader(Config.dataset_path, 'RNE_lgl')
        osm_data.load_data()
        lst_spd = SPD.get_spd_dict(osm_data)

        valid_lst_spd_idxs = np.where(lst_spd > 0) # [array, array]

        len_valid = len(valid_lst_spd_idxs[0])
        len_sampled = int(len_valid * Config.spd_rne_dataset_sampling_rate)
        rand_idx = np.random.choice(len_valid, len_sampled, replace = False) # np.array

        seg_idx_pairs = list(zip(valid_lst_spd_idxs[0][rand_idx], valid_lst_spd_idxs[1][rand_idx])) # [(), (), ...]
        dists = lst_spd[(valid_lst_spd_idxs[0][rand_idx], valid_lst_spd_idxs[1][rand_idx])].tolist() # [], spd

        rtn = {'segidx' : seg_idx_pairs, 'dist' : dists}
        logging.info('[spd_pairs_dataset] @={:.0f}, all.len={}, sampled.len={}' \
                        .format(time.time() - _time, len_valid, len(seg_idx_pairs)))
    
        with open(spd_pairs_file, 'wb') as fh:
            pickle.dump(rtn, fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info('spd_pairs_dataset rtn dumped!!')

    assert Config.spd_rne_dataset_train_partition + Config.spd_rne_dataset_test_partition <= Config.spd_rne_dataset_sampling_rate

    trains_idx = (0, int(len(rtn['segidx']) / Config.spd_rne_dataset_sampling_rate * Config.spd_rne_dataset_train_partition) )
    tests_idx = (int(len(rtn['segidx']) / Config.spd_rne_dataset_sampling_rate * Config.spd_rne_dataset_train_partition), 
                int( len(rtn['segidx']) / Config.spd_rne_dataset_sampling_rate * (Config.spd_rne_dataset_train_partition + Config.spd_rne_dataset_test_partition) ))
    
    logging.info('spd_pairs_dataset. #trains={:d}, #tests={:d}'.format(trains_idx[1]-trains_idx[0], tests_idx[1]-tests_idx[0]))

    return {'trains_segidx' : rtn['segidx'][trains_idx[0]: trains_idx[1]], 
            'trains_dists' : rtn['dist'][trains_idx[0]: trains_idx[1]], 
            'tests_segidx' : rtn['segidx'][tests_idx[0]: tests_idx[1]], 
            'tests_dists' : rtn['dist'][tests_idx[0]: tests_idx[1]]  }
        
