import os
import math
import logging
import random
import time
import pickle
from ast import literal_eval
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_sequence

from config import Config as Config
from utils import tool_funcs
from task.base_task import BaseTask
from task.traj_simi_dataprocess import traj_simi_computation

# V2: backpropogate in every batch
#       test in each epoch

# GRU-based trajectory similarity regression model
class TrajSimiRegression(nn.Module):
    def __init__(self, nfeat, nhidden):
        super(TrajSimiRegression, self).__init__()
        self.rnn = nn.GRU(nfeat, nhidden, num_layers = 2)

    def forward(self, inputs):
        # inputs = [seq, batch_size, nfeat], PackedSequence
        _, h_n = self.rnn(inputs)
        h_n = h_n[-1].squeeze(0) # last layer. size = [batch_size, nhidden]
        return h_n


class TrajSimi(BaseTask):
    def __init__(self, osm_data, embs, encoder):
        super(TrajSimi, self).__init__()
        assert not ( embs == None and encoder == None )

        self.trajsimiregression = None

        self.osm_data = osm_data
        self.seg_ids = self.osm_data.segid_in_adj_segments_graph # list
        self.seg_id_to_idx = self.osm_data.seg_id_to_idx_in_adj_seg_graph
        self.dic_trajsimi = TrajSimi.load_trajsimi_dataset(self.osm_data)

        self.embs = embs
        self.encoder = encoder
        # self.e2e = True if (embs == None and encoder != None) else False
        self.encoder_mode = Config.task_encoder_mode

        self.checkpoint_filepath = '{}/exp/snapshots/{}_trajsimi_{}_best{}.pkl' \
                                    .format(Config.root_dir, Config.dataset_prefix, \
                                            Config.task_encoder_model, Config.dumpfile_uniqueid)


    def train(self):
        # DONOT use model's osm_data, 
        # it may contain extra edges between segments, 
        # e.g. spatial_connections used in SARN.
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_trajsimi start.@={:.3f}, encoder_mode={}".format(training_starttime, self.encoder_mode))

        in_dim = Config.sarn_embedding_dim 
        self.trajsimiregression = TrajSimiRegression(in_dim, Config.trajsimi_rnn_hidden_dim)
        self.trajsimiregression.to(Config.device)
        self.trajsimiregression.train()
        self.criterion = nn.MSELoss()
        self.criterion.to(Config.device)
        self.l1loss = nn.L1Loss(reduction = 'none') # for metric only
        self.l1loss.to(Config.device)
        
        if self.encoder_mode == 'dump':
            optimizer = torch.optim.Adam(self.trajsimiregression.parameters(), \
                                        lr = Config.trajsimi_learning_rate, \
                                        weight_decay = Config.trajsimi_learning_weight_decay)
        elif self.encoder_mode == 'finetune':
            if Config.task_encoder_model == 'SARN_ft':
                optimizer = torch.optim.Adam( \
                                [ {'params': self.trajsimiregression.parameters(), \
                                    'lr': Config.trajsimi_learning_rate, \
                                    'weight_decay': Config.trajsimi_learning_weight_decay}, \
                                    {'params': self.encoder.model.encoder_q.parameters(), \
                                    'lr': Config.trajsimi_learning_rate * Config.task_finetune_lr_rescale} \
                                ] )

        best_loss_eval = 10000000
        best_epoch = 0
        best_mre_eval = 10000000
        bad_counter = 0
        bad_patience = Config.trajsimi_training_bad_patience

        for i_ep in range(Config.trajsimi_epoch):
            _time_ep = time.time()
            train_losses = []
            train_maes = []
            train_mres = []
            gpu_train = []
            ram_train = []

            self.trajsimiregression.train()

            if Config.trajsimi_learning_rated_adjusted:
                if Config.task_encoder_model == 'SARN_ft':
                    _degraded = 0.5 * (1. + math.cos(math.pi * i_ep / Config.trajsimi_epoch))
                    optimizer.param_groups[0]['lr'] = Config.trajsimi_learning_rate * _degraded
                    optimizer.param_groups[1]['lr'] = Config.trajsimi_learning_rate * Config.task_finetune_lr_rescale * _degraded
                else:
                    tool_funcs.adjust_learning_rate(optimizer, Config.trajsimi_learning_rate, \
                                                    i_ep, Config.trajsimi_epoch)

            for i_batch, batch in enumerate( \
                    self.trajsimi_dataset_generator_pairs_batchi(self.dic_trajsimi['trains'], \
                                                                self.dic_trajsimi['trains_simi'], \
                                                                self.dic_trajsimi['max_distance'])):
                _time_batch = time.time()
                sub_seg_idxs, sub_length, sub_embs, sub_simi = batch
                optimizer.zero_grad()
                task_loss, model_loss = 0.0, 0.0

                if self.encoder_mode == 'dump':
                    outs = self.trajsimiregression(sub_embs)
                elif self.encoder_mode == 'finetune':
                    embs = self.encoder.finetune_forward(sub_seg_idxs, True)
                    embs_lst = []
                    _cur_index = 0
                    for l in sub_length:
                        embs_lst.append(embs[_cur_index : _cur_index + l])
                        _cur_index += l
                    sub_embs = pack_sequence(embs_lst)
                    outs = self.trajsimiregression(sub_embs)

                pred_l1_simi = torch.cdist(outs, outs, 1)
                pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]

                loss_train = self.criterion(pred_l1_simi, truth_l1_simi)

                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())
                gpu_train.append(tool_funcs.GPUInfo.mem()[0])
                ram_train.append(tool_funcs.RAMInfo.mem())

                with torch.no_grad():
                    mae = self.l1loss(pred_l1_simi, truth_l1_simi)
                    mre = torch.mean(mae / truth_l1_simi)

                    train_maes.append( torch.mean(mae).item() )
                    train_mres.append( torch.mean(mre).item() )

                # debug output
                if i_batch % 100 == 0:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}({:.4f}/{:.4f}), "
                                "mre={:.3f}, mae={:.3f}, @={:.3f}, gpu={}, ram={}" \
                                .format(i_ep, i_batch, loss_train.item(), task_loss, model_loss,
                                        train_mres[-1], train_maes[-1],
                                        time.time()-_time_batch, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            # i_ep
            logging.info("training. i_ep={}, loss={:.4f}, mre={:.3f}, mae={:.3f}, @={:.3f}" \
                        .format(i_ep, sum(train_losses)/len(train_losses), 
                                sum(train_mres) / len(train_mres), 
                                sum(train_maes) / len(train_maes),
                                time.time()-_time_ep))
            
            eval_metrics = self.test(self.dic_trajsimi['evals'], \
                                        self.dic_trajsimi['evals_simi'], \
                                        self.dic_trajsimi['max_distance'])
            logging.info("eval.     i_ep={}, loss={:.4f}, mre={:.3f}, mae={:.3f}, hr={:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            loss_eval_ep = eval_metrics[0]
            mre_eval_ep = eval_metrics[1]

            training_gpu_usage = tool_funcs.mean(gpu_train)
            training_ram_usage = tool_funcs.mean(ram_train)

            # early stopping
            if  loss_eval_ep < best_loss_eval:
                best_epoch = i_ep
                best_loss_eval = loss_eval_ep
                best_mre_eval = mre_eval_ep
                bad_counter = 0
                
                if self.encoder_mode == 'finetune':
                    if Config.task_encoder_model == 'SARN_ft':
                        torch.save({ "encoder.feat_emb" : self.encoder.feat_emb.state_dict(),
                                    "encoder.encoder_q" : self.encoder.model.encoder_q.state_dict(),
                                    "trajsimi": self.trajsimiregression.state_dict()}, 
                                    self.checkpoint_filepath)
                else:
                    torch.save({'trajsimi': self.trajsimiregression.state_dict()}, 
                                self.checkpoint_filepath)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == Config.trajsimi_epoch:
                training_endtime = time.time()
                logging.info("training end. @={:.3f}, best_epoch={}, best_loss_eval={:.4f}, best_mre_eval={:.3f}" \
                                .format(training_endtime - training_starttime, \
                                        best_epoch, best_loss_eval, best_mre_eval))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_filepath)
        self.trajsimiregression.load_state_dict(checkpoint['trajsimi'])
        self.trajsimiregression.to(Config.device)
        self.trajsimiregression.eval()

        if self.encoder_mode == 'finetune':
            if Config.task_encoder_model == 'SARN_ft':
                self.encoder.feat_emb.load_state_dict(checkpoint['encoder.feat_emb'])
                self.encoder.model.encoder_q.load_state_dict(checkpoint['encoder.encoder_q'])

        test_starttime = time.time()
        _test_metrics = self.test(self.dic_trajsimi['tests'], \
                                    self.dic_trajsimi['tests_simi'], \
                                    self.dic_trajsimi['max_distance'])
        test_endtime = time.time()
        logging.info("test. @={:.3f}, loss={:.4f}, mre={:.3f}, mae={:.3f}, " \
                    "hr={:.3f},{:.3f},{:.3f}".format( \
                    test_endtime - test_starttime, *_test_metrics))

        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': _test_metrics[6], \
                'task_test_ram': _test_metrics[7], \
                'hr5':_test_metrics[3], 'hr20':_test_metrics[4], 'hr20in5':_test_metrics[5]}


    @torch.no_grad()
    def test(self, datasets, datasets_simi, max_distance):
        _time = time.time()

        self.trajsimiregression.eval()
        
        # dont use trajsimi_dataset_generator_pairs_batchi
        # logic here is different from training
        datasets_simi = torch.tensor(datasets_simi, device = Config.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        traj_embs = []

        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(datasets)):
            sub_seg_idxs, sub_length, sub_embs = batch
            if self.encoder_mode == 'dump':
                outs = self.trajsimiregression(sub_embs)
            elif self.encoder_mode == 'finetune':
                embs = self.encoder.finetune_forward(sub_seg_idxs, False)
                embs_lst = []
                _cur_index = 0
                for l in sub_length:
                    embs_lst.append(embs[_cur_index : _cur_index + l])
                    _cur_index += l
                sub_embs = pack_sequence(embs_lst, False)
                outs = self.trajsimiregression(sub_embs)

            traj_embs.append(outs)

        traj_embs = torch.cat(traj_embs)
        pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        # metrics
        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)
        mae = self.l1loss(pred_l1_simi_seq, truth_l1_simi_seq)
        mre = torch.mean(mae / truth_l1_simi_seq)

        hrA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hrB = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
        hrBinA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)

        return loss.item(), torch.mean(mre).item(), torch.mean(mae).item(), \
                hrA, hrB, hrBinA, \
                tool_funcs.GPUInfo.mem()[0], tool_funcs.RAMInfo.mem()
                

    @torch.no_grad()
    def trajsimi_dataset_generator_batchi(self, datasets):
        cur_index = 0
        len_datasets = len(datasets)

        while cur_index < len_datasets:
            end_index = cur_index + Config.trajsimi_batch_size \
                                if cur_index + Config.trajsimi_batch_size < len_datasets \
                                else len_datasets

            sub_embs = []
            sub_seg_idxs = []
            sub_length = []
            for d_idx in range(cur_index, end_index):
                traj_id, traj = datasets[d_idx]
                seg_idxs = list(map(lambda seg_id: self.seg_id_to_idx[seg_id], traj))
                if self.encoder_mode == 'dump':
                    emb = self.embs[seg_idxs]
                    sub_embs.append(emb)
                else:
                    sub_seg_idxs += seg_idxs
                    sub_length.append(len(seg_idxs))

            if self.encoder_mode == 'dump':
                sub_embs = pack_sequence(sub_embs, False)

            yield sub_seg_idxs, sub_length, sub_embs
            cur_index = end_index


    # It is not an exact batchy data generator. 
    # we estimate how many batches could be in one epoch
    # , since each data in a batchy is an independtly random event
    def trajsimi_dataset_generator_pairs_batchi(self, datasets, datasets_simi, max_distance):
        len_datasets = len(datasets)
        datasets_simi = torch.tensor(datasets_simi, device = Config.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        
        count_i = 0
        batch_size = len_datasets if len_datasets < Config.trajsimi_batch_size else Config.trajsimi_batch_size
        counts = math.ceil( (len_datasets / batch_size)**2 )

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k = batch_size)
            dataset_idxs_sample.sort(key = lambda idx: len(datasets[idx][1]), reverse = True) # len descending order

            sub_embs = []
            sub_seg_idxs = []
            sub_length = []
            for d_idx in dataset_idxs_sample:
                traj_id, traj = datasets[d_idx]
                seg_idxs = list(map(lambda seg_id: self.seg_id_to_idx[seg_id], traj))
                if self.encoder_mode == 'dump':
                    emb = self.embs[seg_idxs]
                    sub_embs.append(emb)
                else:
                    sub_seg_idxs += seg_idxs
                    sub_length.append(len(seg_idxs))

            if self.encoder_mode == 'dump':
                sub_embs = pack_sequence(sub_embs)
            sub_simi = datasets_simi[dataset_idxs_sample][:,dataset_idxs_sample]

            yield sub_seg_idxs, sub_length, sub_embs, sub_simi
            count_i += 1

    @staticmethod
    def hitting_ratio(preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int):
        # hitting ratio: see DiYao R10@50 metric
        # the overlap percentage of the topk predicted results and the topk ground truth
        # overlap(overlap(preds@pred_topk, truths@truth_topk), truths@truth_topk) / truth_topk

        # preds = [batch_size, class_num], tensor, element indicates the probability
        # truths = [batch_size, class_num], tensor, element indicates the probability
        assert preds.shape == truths.shape and pred_topk < preds.shape[1] and truth_topk < preds.shape[1]
    
        _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim = 1, largest = False)
        _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim = 1, largest = False)

        preds_k_idx = preds_k_idx.cpu()
        truths_k_idx = truths_k_idx.cpu()

        tp = sum([np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size for i in range(preds_k_idx.shape[0])])
        
        return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])


    @staticmethod
    def load_trajsimi_dataset(osm_data):
        # pre-computed traj similarity matrix
        dic_trajsimi = None
        _dic_trajsimi_file = '{}/data/{}_traj_simi_dict.pickle'.format(Config.root_dir, Config.trajsimi_prefix)
        if os.path.exists(_dic_trajsimi_file):
            with open(_dic_trajsimi_file, 'rb') as fh:
                dic_trajsimi = pickle.load(fh)
        else:
            _traj_file = '{}/data/{}'.format(Config.root_dir, Config.trajsimi_prefix)
            pd_trajs = pd.read_csv(_traj_file, delimiter = ',', index_col = 'traj_id')
            pd_trajs.loc[:, 'mm_edges'] = pd_trajs.mm_edges.apply(literal_eval)
            dic_trajs = pd_trajs[['mm_edges']].to_dict('index') # {traj_id -> {mm_edges: []}, {} ...}

            dic_trajsimi = traj_simi_computation(osm_data, dic_trajs)

        logging.info("traj dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(dic_trajsimi['trains']), len(dic_trajsimi['evals']), len(dic_trajsimi['tests'])))
        return dic_trajsimi

