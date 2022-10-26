import sys
sys.path.append('..')

import logging
import time
import copy
import random
import typing
import math
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pickle
from scipy.sparse import coo_matrix

from config import Config as Config
from utils import tool_funcs
from utils.osm_loader import OSMLoader
from utils.osm_loader import EdgeIndex
from utils.edge_index import EdgeIndex
from model.base import BaseEncoder
from model.moco_multi import MoCo as MoCoMultiQ
from model.feat_embedding import FeatEmbedding


class SARN(BaseEncoder):
    def __init__(self):
        self.osm_data = OSMLoader(Config.dataset_path, schema = 'SARN')
        self.osm_data.load_data()
        
        self.seg_feats = self.osm_data.seg_feats
        self.checkpoint_path = '{}/exp/snapshots/{}_SARN_best{}.pkl'.format(Config.root_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)
        self.embs_path = '{}/exp/snapshots/{}_SARN_best_embs{}.pickle'.format(Config.root_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)
        
        self.feat_emb = FeatEmbedding(self.osm_data.count_wayid_code,
                                    self.osm_data.count_segid_code,
                                    self.osm_data.count_highway_cls,
                                    self.osm_data.count_length_code,
                                    self.osm_data.count_radian_code,
                                    self.osm_data.count_s_lon_code,
                                    self.osm_data.count_s_lat_code).to(Config.device)

        self.model = MoCoMultiQ(nfeat = Config.sarn_seg_feat_dim,
                                nemb = Config.sarn_embedding_dim, 
                                nout = Config.sarn_out_dim,
                                queue_size = Config.sarn_moco_each_queue_size,
                                nqueue = self.osm_data.cellspace.lon_size * self.osm_data.cellspace.lat_size,
                                temperature = Config.sarn_moco_temperature).to(Config.device)

        logging.info('[Moco] total_queue_size={:.0f}, multi_side_length={:.0f}, '
                        'multi_nqueues={}*{}, each_queue_size={}, real_total={}, local_weight={:.2f}' \
                    .format(Config.sarn_moco_total_queue_size, \
                            Config.sarn_moco_multi_queue_cellsidelen, \
                            self.osm_data.cellspace.lon_size, \
                            self.osm_data.cellspace.lat_size, \
                            Config.sarn_moco_each_queue_size, \
                            Config.sarn_moco_each_queue_size * self.osm_data.cellspace.lon_size * self.osm_data.cellspace.lat_size, \
                            Config.sarn_moco_loss_local_weight))
        
        if Config.task_encoder_mode == 'finetune':
            self.t_edge_index = copy.deepcopy(self.osm_data.edge_index.edges.T)
            self.t_edge_index = torch.tensor(self.t_edge_index, dtype = torch.long, device = Config.device)

        self.seg_id_to_idx = self.osm_data.seg_id_to_idx_in_adj_seg_graph
        self.seg_idx_to_id = self.osm_data.seg_idx_to_id_in_adj_seg_graph
        self.seg_id_to_cellid = dict(self.osm_data.segments.reset_index()[['inc_id','c_cellid']].values.tolist()) # contains those not legal segments
        self.seg_idx_to_cellid = [-1] * len(self.seg_idx_to_id)
        for _id, _cellid in self.seg_id_to_cellid.items():
            _idx = self.seg_id_to_idx.get(_id, -1)
            if _idx >= 0:
                self.seg_idx_to_cellid[_idx] = _cellid
        assert sum(filter(lambda x: x < 0, self.seg_idx_to_cellid)) == 0


    def train(self):
        # init model, loss, ...
        # create adj
        # 1. each epoch
        #   generate augmented graph (adj) 
        # 2. each batch
        #   select a batch nodes in whole graph
        #   generate sub_adj, then sub_adj -> sub_edge_index
        #   train

        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.feat_emb.parameters()), 
                                        lr = Config.sarn_learning_rate,
                                        weight_decay = Config.sarn_learning_weight_decay)
        
        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.sarn_training_bad_patience

        for i_ep in range(Config.sarn_epochs):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            self.feat_emb.train()
            self.model.train()

            if Config.sarn_learning_rated_adjusted:
                tool_funcs.adjust_learning_rate(optimizer, Config.sarn_learning_rate, i_ep, Config.sarn_epochs)
            
            # drop edges from edge_index
            edge_index_1 = copy.deepcopy(self.osm_data.edge_index)
            edge_index_1 = graph_aug_edgeindex(edge_index_1)

            edge_index_2 = copy.deepcopy(self.osm_data.edge_index)
            edge_index_2 = graph_aug_edgeindex(edge_index_2)

            for i_batch, batch in enumerate(self.__train_data_generator_batchi(edge_index_1, edge_index_2, shuffle = True)):
                _time_batch = time.time()
                
                optimizer.zero_grad()
                (sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1), \
                        (sub_seg_feats_2, sub_edge_index_2, mapping_to_origin_idx_2), \
                        sub_seg_ids, sub_cellids = batch

                sub_seg_feats_1 = self.feat_emb(sub_seg_feats_1)
                sub_seg_feats_2 = self.feat_emb(sub_seg_feats_2)

                model_rtn = self.model(sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1, \
                                        sub_seg_feats_2, sub_edge_index_2, mapping_to_origin_idx_2, \
                                        sub_cellids, sub_seg_ids)
                loss = self.model.loss_mtl(*model_rtn, Config.sarn_moco_loss_local_weight, Config.sarn_moco_loss_global_weight)

                loss.backward()
                optimizer.step()
                loss_ep.append(loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                if i_batch % 50 == 0:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                            .format(i_ep, i_batch, loss.item(), time.time() - _time_batch, \
                                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))


            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                    .format(i_ep, loss_ep_avg, time.time() - _time_ep, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # early stopping
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                torch.save({'model_state_dict': self.model.state_dict(),
                            'feat_emb_state_dict': self.feat_emb.state_dict()},
                            self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == Config.sarn_epochs:
                logging.info("[Training] END! best_epoch={}, best_loss_train={:.6f}" \
                            .format(best_epoch, best_loss_train))
                break
        
        return {'enc_train_time': time.time()-training_starttime, \
                'enc_train_gpu': training_gpu_usage, \
                'enc_train_ram': training_ram_usage}


    def test(self):
        pass


    def finetune_forward(self, sub_seg_idxs, is_training: bool):
        if is_training:
            self.feat_emb.train()
            self.model.train()
            embs = self.model.encoder_q(self.feat_emb(self.seg_feats), self.t_edge_index)[sub_seg_idxs]

        else:
            with torch.no_grad():
                self.feat_emb.eval()
                self.model.eval()
                embs = self.model.encoder_q(self.feat_emb(self.seg_feats), self.t_edge_index)[sub_seg_idxs]
        return embs


    def __train_data_generator_batchi(self, edge_index_1: EdgeIndex, \
                                            edge_index_2: typing.Union[EdgeIndex, None], \
                                            shuffle = True):
        cur_index = 0
        n_segs = len(self.seg_idx_to_id)
        seg_idxs = list(range(n_segs))

        if shuffle: # for training
            random.shuffle(seg_idxs)

        while cur_index < n_segs:
            end_index = cur_index + Config.sarn_batch_size \
                            if cur_index + Config.sarn_batch_size < n_segs \
                            else n_segs
            sub_seg_idx = seg_idxs[cur_index: end_index]

            sub_edge_index_1, new_x_idx_1, mapping_to_origin_idx_1 = \
                                edge_index_1.sub_edge_index(sub_seg_idx)
            sub_seg_feats_1 = self.seg_feats[new_x_idx_1]
            sub_edge_index_1 = torch.tensor(sub_edge_index_1, dtype = torch.long, device = Config.device)
            
            if edge_index_2 != None:
                sub_edge_index_2, new_x_idx_2, mapping_to_origin_idx_2 = \
                                    edge_index_2.sub_edge_index(sub_seg_idx)
                sub_seg_feats_2 = self.seg_feats[new_x_idx_2]
                sub_edge_index_2 = torch.tensor(sub_edge_index_2, dtype = torch.long, device = Config.device)

                sub_seg_ids = [self.seg_idx_to_id[idx] for idx in sub_seg_idx]
                sub_cellids = [self.seg_idx_to_cellid[idx] for idx in sub_seg_idx]
                sub_seg_ids = torch.tensor(sub_seg_ids, dtype = torch.long, device = Config.device)
                sub_cellids = torch.tensor(sub_cellids, dtype = torch.long, device = Config.device)
                
                yield (sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1), \
                        (sub_seg_feats_2, sub_edge_index_2, mapping_to_origin_idx_2), \
                        sub_seg_ids, sub_cellids
            else:
                yield sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1

            cur_index = end_index
    

    @torch.no_grad()
    def load_model_state(self, f_path):
        checkpoint = torch.load(f_path)
        self.feat_emb.load_state_dict(checkpoint['feat_emb_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feat_emb.to(Config.device)
        self.model.to(Config.device)


    @torch.no_grad()
    def get_embeddings(self, from_checkpoint): # return embs on cpu!
        if from_checkpoint:
            self.load_model_state(self.checkpoint_path)

        edge_index_1 = copy.deepcopy(self.osm_data.edge_index)

        self.feat_emb.eval()
        self.model.eval()
        embs = torch.empty((0), device = Config.device)

        with torch.no_grad():
            for i_batch, batch in enumerate(self.__train_data_generator_batchi(edge_index_1, None, shuffle = False)):
                    
                sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1 = batch
                sub_seg_feats_1 = self.feat_emb(sub_seg_feats_1)

                emb = self.model.encoder_q(sub_seg_feats_1, sub_edge_index_1)
                emb = emb[mapping_to_origin_idx_1]
                embs = torch.cat((embs, emb), 0)

            embs = F.normalize(embs, dim = 1) # dim=0 feature norm, dim=1 obj norm
            return embs
        return None


    @torch.no_grad()
    def dump_embeddings(self, embs = None):
        if embs == None:
            embs = self.get_embeddings(True)
        with open(self.embs_path, 'wb') as fh:
            pickle.dump(embs, fh, protocol = pickle.HIGHEST_PROTOCOL)
        logging.info('[dump embedding] done.')
        return


def graph_aug_edgeindex(edge_index: EdgeIndex):
    # 1. sample to-be-removed edges by weights respectively
    # 2. union all these to-be-removed edges in one set

    _time = time.time()
    n_ori_edges = edge_index.length()
    n_topo_remove = n_spatial_remove = 0

    edges_topo_weight = edge_index.tweight # shallow copy
    edges_topo_weight_0 = (edges_topo_weight == 0) # to mask
    n_topo = n_ori_edges - sum(edges_topo_weight_0)

    max_tweight = max(edges_topo_weight) + 1.5
    edges_topo_weight = np.log(max_tweight - edges_topo_weight) / np.log(1.5)
    edges_topo_weight[edges_topo_weight_0] = 0
    sum_tweight = sum(edges_topo_weight)
    edges_topo_weight = edges_topo_weight / sum_tweight
    edges_topo_weight = edges_topo_weight.tolist()

    edges_idxs_to_remove = set(np.random.choice(n_ori_edges, p = edges_topo_weight, \
                                                size = int(Config.sarn_break_edge_topo_prob * n_topo), \
                                                replace = False))
    n_topo_remove = len(edges_idxs_to_remove)

    edges_idxs_to_remove = list(edges_idxs_to_remove)
    edge_index.remove_edges(edges_idxs_to_remove)

    logging.debug("[Graph Augment] @={:.0f}, #original_edges={}, #edges_broken={} ({}+{}), #edges_left={}" \
                    .format(time.time() - _time, n_ori_edges, len(edges_idxs_to_remove), \
                            n_topo_remove, n_spatial_remove, edge_index.length() ))

    return edge_index



