# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/moco
# https://arxiv.org/abs/1911.05722

# Modified by yc - 2021
# version 3:
#   add multiple negative sampling queues
#   add local-global loss
#   optimise the process of pos/neg pair computation - speedup 3x
#   adapted for GAT

from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GAT_pyG import GAT


class MoCo(nn.Module):
    def __init__(self, nfeat, nemb, nout,
                queue_size, nqueue, mmt = 0.999, temperature = 0.07):
        super(MoCo, self).__init__()

        self.queue_size = queue_size
        self.nqueue = nqueue
        self.mmt = mmt
        self.temperature = temperature

        # for simplicity, initialize gat objects here
        self.encoder_q = GAT(nfeat = nfeat, nhid = nfeat // 2, nout = nemb, nhead = 4, nlayer = 2)
        self.encoder_k = GAT(nfeat = nfeat, nhid = nfeat // 2, nout = nemb, nhead = 4, nlayer = 2)

        self.mlp_q = Projector(nemb, nemb, nout) 
        self.mlp_k = Projector(nemb, nemb, nout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queues
        self.queues = MomentumQueue(nout, queue_size, nqueue)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)
        
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)


    def forward(self, inputs_q, edge_index_q, idx_in_adjsub_q, 
                        inputs_k, edge_index_k, idx_in_adjsub_k,
                        q_ids, elem_ids):
        # length of different parameteres may be different

        # compute query features
        q = self.mlp_q(self.encoder_q(inputs_q, edge_index_q)) # q: [?, nfeat]
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.mlp_k(self.encoder_k(inputs_k, edge_index_k))  
            k = nn.functional.normalize(k, dim=1)

        q = q[idx_in_adjsub_q] # q: [batch, nfeat]
        k = k[idx_in_adjsub_k] # k: [batch, nfeat]

        # positive logits
        l_pos_local = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # [batch, 1]

        neg_local = self.queues.queue[q_ids].clone().detach() # [batch, nfeat, queue_size]
        l_neg_local = torch.einsum('nc,nck->nk', q, neg_local) # [batch, queue_size]

        neg_local_ids = self.queues.ids[q_ids].clone().detach() # [batch, queue_size]
        l_neg_local[neg_local_ids == elem_ids.unsqueeze(1).repeat(1, neg_local_ids.shape[1])] = -9e15  # [batch, queue_size]

        neg_global = torch.mean(self.queues.queue.clone().detach(), dim = 2) # [nqueue, nfeat], readout
        l_neg_global = torch.einsum('nc,ck->nk', q, neg_global.T) # [batch, nqueue]

        # logits: 
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) # [batch, (1+queue_size)]
        logits_global = l_neg_global

        # apply temperature
        logits_local /= self.temperature
        logits_global /= self.temperature

        # labels: positive key indicators
        labels_local = torch.zeros_like(l_pos_local, dtype = torch.long).squeeze(1)
        labels_global = q_ids.clone().detach()

        # dequeue and enqueue
        for i, q_id in enumerate(q_ids):
            self.queues.dequeue_and_enqueue(k[i,:], elem_ids[i].item(), q_id)

        return logits_local, labels_local, logits_global, labels_global


    # local and global losses are served as a multi-task-learning loss
    def loss_mtl(self, logits_local, labels_local, logits_global, labels_global,
                w_local, w_global):
        
        # temperature has applied in forward()
        sfmax_local = F.softmax(logits_local, dim = 1) # [batch, 1+queue_size]
        sfmax_global = F.softmax(logits_global, dim = 1) # [batch, n_queue]
        p_local = torch.log(
                    sfmax_local.gather(1, labels_local.view(-1,1))) # [batch, 1]
        p_global = torch.log(
                    sfmax_global.gather(1, labels_global.view(-1,1))) # [batch, 1]

        loss_local = F.nll_loss(p_local, torch.zeros_like(labels_local))
        loss_global = F.nll_loss(p_global, torch.zeros_like(labels_local)) 

        return loss_local * w_local + loss_global * w_global


class Projector(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(nin, nhid), 
                                nn.ReLU(), 
                                nn.Linear(nhid, nout))
        self.reset_parameter()

    def forward(self, x):
        return self.mlp(x)

    def reset_parameter(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.414)
                torch.nn.init.zeros_(m.bias)
        
        self.mlp.apply(_weights_init)
        

# multi queue
class MomentumQueue(nn.Module):

    def __init__(self, nhid, queue_size, nqueue):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.nqueue =  nqueue

        self.register_buffer("queue", torch.randn(nqueue, nhid, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim = 1)

        self.register_buffer("ids", torch.full([nqueue, queue_size], -1, dtype = torch.long))

        self.register_buffer("queue_ptr", torch.zeros((nqueue), dtype = torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, k, elem_id, q_id):
        # k: feature
        # elem_id: used in ids
        # q_id: queue id

        ptr = int(self.queue_ptr[q_id].item())
        self.queue[q_id, :, ptr] = k.T
        self.ids[q_id, ptr] = elem_id

        ptr = (ptr + 1) % self.queue_size
        self.queue_ptr[q_id] = ptr

