# label prediction - segment classification

import logging
import time
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config as Config
from utils import tool_funcs
from task.base_task import BaseTask


class FCNClassifier(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(FCNClassifier, self).__init__()
        
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)

    def forward(self, inputs):
        # input = [batch_size * nhid]  ps. batch_size is #segments here
        out = F.dropout(inputs, 0.2, training = self.training)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out # [batch_size * nclass]


class Classifier(BaseTask):

    def __init__(self, osm_data, embs, encoder):
        super(Classifier, self).__init__()
        assert not ( embs == None and encoder == None)

        self.classifier = None

        self.osm_data = osm_data
        self.embs = embs
        self.encoder = encoder
        self.encoder_mode = Config.task_encoder_mode

        self.checkpoint_filepath = '{}/exp/snapshots/{}_classifier_{}_best{}.pkl'.format( \
                                    Config.root_dir, Config.dataset_prefix, \
                                    Config.task_encoder_model, Config.dumpfile_uniqueid)


    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_classifier start.@={:.3f}, encoder_mode={}".format(
                    training_starttime, self.encoder_mode))
        
        train_dataset_idx, train_dataset_label, \
                eval_dataset_idx, eval_dataset_label, \
                test_dataset_idx, test_dataset_label, \
                class_counts = self.classifier_datasets(Config.fcnclassifier_classify_colname)

        in_dim = Config.sarn_embedding_dim
        self.classifier = FCNClassifier(in_dim, Config.fcnclassifier_nhidden, \
                                        len(class_counts)).to(Config.device)
        self.classifier.to(Config.device)
        self.classifier.train()
        self.criterion = nn.CrossEntropyLoss().to(Config.device)
        self.criterion.to(Config.device)
        
        if self.encoder_mode == 'dump':
            optimizer = torch.optim.Adam(self.classifier.parameters(), \
                                    lr = Config.fcnclassifier_learning_rate, \
                                    weight_decay = Config.fcnclassifier_learning_weight_decay)
        elif self.encoder_mode == 'finetune':
            if Config.task_encoder_model == 'SARN_ft':
                optimizer = torch.optim.Adam( \
                                        [ {'params': self.classifier.parameters(), \
                                            'lr': Config.fcnclassifier_learning_rate, \
                                            'weight_decay': Config.fcnclassifier_learning_weight_decay}, \
                                          {'params': self.encoder.model.encoder_q.parameters(), \
                                            'lr': Config.fcnclassifier_learning_rate * Config.task_finetune_lr_rescale} \
                                        ])

        best_loss_eval = 100000
        best_epoch = 0
        best_f1_eval = 0
        bad_counter = 0
        bad_patience = Config.fcnclassifier_training_bad_patience

        for i_ep in range(Config.fcnclassifier_epoch):
            losses_train = []
            f1s_train = []
            gpu_train = []
            ram_train = []

            self.classifier.train()

            for i_batch, batch in enumerate( \
                    self.classifier_dataset_generator_batchi(train_dataset_idx, train_dataset_label)):

                dataset_idx_batch, dataset_emb_batch, dataset_label_batch = batch
                optimizer.zero_grad()
                task_loss, model_loss = 0.0, 0.0

                if self.encoder_mode == 'dump':
                    pred_train = self.classifier(dataset_emb_batch)
                elif self.encoder_mode == 'finetune':
                    pred_train = self.classifier(self.encoder.finetune_forward(dataset_idx_batch, True))

                loss_train = self.criterion(pred_train, dataset_label_batch)

                loss_train.backward()
                optimizer.step()

                pred_train = torch.argmax(pred_train, 1).tolist()
                gtruth_train = dataset_label_batch.tolist()
                
                losses_train.append(loss_train.item())
                f1s_train.append(tool_funcs.f1(gtruth_train, pred_train))

                gpu_train.append(tool_funcs.GPUInfo.mem()[0])
                ram_train.append(tool_funcs.RAMInfo.mem())

            loss_eval, f1_eval, _, _, _ = self.test(eval_dataset_idx, eval_dataset_label, class_counts)

            if i_ep % 20 == 0:
                logging.info("training. ep={}, loss_train={:.4f}({:.4f}/{:.4f}), "
                            "f1_train={:.4f}, loss_eval={:.4f}, f1_eval={:.4f}, gpu={}, ram={}" \
                            .format(i_ep, tool_funcs.mean(losses_train), task_loss, model_loss, \
                                    tool_funcs.mean(f1s_train), \
                                    loss_eval, f1_eval,
                                    tool_funcs.mean(gpu_train), tool_funcs.mean(ram_train)))

            training_gpu_usage = tool_funcs.mean(gpu_train)
            training_ram_usage = tool_funcs.mean(ram_train)

            # early stopping
            if loss_eval < best_loss_eval:
                best_epoch = i_ep
                best_loss_eval = loss_eval
                best_f1_eval = f1_eval
                bad_counter = 0
                if self.encoder_mode == 'dump':
                    torch.save({"classifier": self.classifier.state_dict()}, 
                                self.checkpoint_filepath)
                elif self.encoder_mode == 'finetune':
                    if Config.task_encoder_model == 'SARN_ft':
                        torch.save({"encoder.feat_emb" : self.encoder.feat_emb.state_dict(),
                                    "encoder.encoder_q" : self.encoder.model.encoder_q.state_dict(),
                                    "classifier": self.classifier.state_dict()}, 
                                    self.checkpoint_filepath)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == Config.fcnclassifier_epoch:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_eval={:.4f}, best_f1_eval={:.4f}" \
                            .format(training_endtime - training_starttime, best_epoch, best_loss_eval, best_f1_eval))
                break
        
        checkpoint = torch.load(self.checkpoint_filepath)
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.classifier.to(Config.device)
        self.classifier.eval()
        if self.encoder_mode == 'finetune':
            if Config.task_encoder_model == 'SARN_ft':
                self.encoder.feat_emb.load_state_dict(checkpoint['encoder.feat_emb'])
                self.encoder.model.encoder_q.load_state_dict(checkpoint['encoder.encoder_q'])

        test_starttime = time.time()
        _, f1_test, auc_test, gpu_test, ram_test = self.test(test_dataset_idx, test_dataset_label, class_counts)
        test_endtime = time.time()
        logging.info("test. @={:.3f}, f1={:.6f}, auc={:.6f}".format(test_endtime - test_starttime, f1_test, auc_test) )

        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': gpu_test, \
                'task_test_ram': ram_test, \
                'f1': f1_test, 'auc': auc_test}


    @torch.no_grad()
    def test(self, dataset_idx, dataset_label, class_counts):
        losses= []
        f1s = []
        aucs = []
        gpus = []
        rams = []
        
        self.classifier.eval()

        for i_batch, batch in enumerate(self.classifier_dataset_generator_batchi(dataset_idx, dataset_label)):
            dataset_idx_batch, dataset_emb_batch, dataset_label_batch = batch

            if self.encoder_mode == 'dump':
                pred = self.classifier(dataset_emb_batch)
            elif self.encoder_mode == 'finetune':
                pred = self.classifier(self.encoder.finetune_forward(dataset_idx_batch, False))

            loss = self.criterion(pred, dataset_label_batch)
            pred = torch.argmax(pred, 1).tolist()
            gtruth = dataset_label_batch.tolist()
            
            losses.append(loss.item())
            f1s.append(tool_funcs.f1(gtruth, pred))
            aucs.append(tool_funcs.auc(gtruth, pred, sorted(class_counts.keys()) ))

            gpus.append(tool_funcs.GPUInfo.mem()[0])
            rams.append(tool_funcs.RAMInfo.mem())

        return tool_funcs.mean(losses), tool_funcs.mean(f1s), tool_funcs.mean(aucs), \
                tool_funcs.mean(gpus), tool_funcs.mean(rams)


    def classifier_datasets(self, type_colname):
        seg_ids = list(self.osm_data.segid_in_adj_segments_graph)
        seg_id_to_idx = self.osm_data.seg_id_to_idx_in_adj_seg_graph

        if type_colname == 'maxspeed':
            ms = self.osm_data.segments.reset_index()[['inc_id', 'maxspeed']]
            ms = ms[ms.maxspeed != 'NG'] # remove NG type, only segments having speed limits are preserved.
            ms['maxspeed'], ms_types = pd.factorize(ms.maxspeed) # maxspeed type starts from 0
            class_counts = ms.maxspeed.value_counts().to_dict() # {0: 33, 1: 11, ...}
            
            lst_idx_ms = ms.reset_index()[['inc_id','maxspeed']].values.tolist()
            lst_idx_ms = list(filter(lambda x: x[0] in seg_ids, lst_idx_ms))
            lst_idx_ms = [ [seg_id_to_idx[x[0]], x[1]] for x in lst_idx_ms ] # ['idx_ein_emb','ms_type']
            dataset = lst_idx_ms
            random.shuffle(dataset)
            logging.debug("classify maxspeed-> #dataset={}, ms_types={}".format(len(dataset), class_counts))

        len_dataset = len(dataset)

        # split into training, evaluation and test sets
        train_dataset = dataset[ : int(len_dataset * 0.6)]
        train_dataset_idx = [x[0] for x in train_dataset]
        train_dataset_label = torch.tensor([x[1] for x in train_dataset], dtype = torch.long, device = Config.device)
        
        eval_dataset = dataset[ int(len_dataset * 0.6) : int(len_dataset * 0.8)]
        eval_dataset_idx = [x[0] for x in eval_dataset]
        eval_dataset_label = torch.tensor([x[1] for x in eval_dataset], dtype = torch.long, device = Config.device)

        test_dataset = dataset[int(len_dataset * 0.8): ]
        test_dataset_idx = [x[0] for x in test_dataset]
        test_dataset_label = torch.tensor([x[1] for x in test_dataset], dtype = torch.long, device = Config.device)

        logging.info("classify -> #train={}, #eval={}, #test={}".format( \
                    len(train_dataset), len(eval_dataset), len(test_dataset)))
        
        return train_dataset_idx, train_dataset_label, \
                eval_dataset_idx, eval_dataset_label, \
                test_dataset_idx, test_dataset_label, \
                class_counts


    def classifier_dataset_generator_batchi(self, dataset_idx, dataset_label):
        cur_index = 0
        len_dataset = len(dataset_idx)

        while cur_index < len_dataset:
            end_index = cur_index + Config.fcnclassifier_batch_size \
                                if cur_index + Config.fcnclassifier_batch_size < len_dataset \
                                else len_dataset          
            if self.encoder_mode == 'finetune':
                dataset_idx_batch = dataset_idx[cur_index: end_index]
                dataset_emb_batch = None
                dataset_label_batch = dataset_label[ cur_index: end_index ]
            else:
                dataset_idx_batch = None
                dataset_emb_batch = self.embs[ dataset_idx[cur_index: end_index] ]
                dataset_label_batch = dataset_label[ cur_index: end_index ]

            yield dataset_idx_batch, dataset_emb_batch, dataset_label_batch

            cur_index += Config.fcnclassifier_batch_size