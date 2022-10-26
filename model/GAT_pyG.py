# copyright: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, nhead, nlayer = 1):
        super(GAT, self).__init__()
        assert nlayer >= 1

        self.nlayer = nlayer
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(nfeat, nhid, heads = nhead, 
                                    dropout = 0.2, negative_slope = 0.2))
        for _ in range(nlayer - 1):
            self.layers.append(GATConv(nhid * nhead, nhid, heads = nhead, 
                                    dropout = 0.2, negative_slope = 0.2))
        self.layer_out = GATConv(nhead * nhid, nout, heads = 1, concat=False,
                            dropout = 0.2, negative_slope = 0.2)

    # x = [2708, 1433]
    # edge_index = [2, 10556], pair-wise adj edges
    def forward(self, x, edge_index):

        for l in range(self.nlayer):
            x = F.dropout(x, p = 0.2, training = self.training)
            x = self.layers[l](x, edge_index)
            x = F.elu(x)
        # output projection
        x = F.dropout(x, p = 0.2, training = self.training)
        x = self.layer_out(x, edge_index)

        return x


# adj = [N, N] : tensor/sparse tensor
# idx = [B] [nodeidx1, nodeidx2, ....] : tensor, 
#       idx can have duplicated items and unordered!
def create_sub_adj(adj_mat, idx, on_device, out_device):

    sorted_idx = torch.unique(idx, sorted = True)
    
    # we omit prefix 'sorted_' for simplicity
    N = adj_mat.shape[0]

    idx_mask = torch.zeros(N, dtype = torch.long, device = on_device)
    idx_mask.index_fill_(0, sorted_idx, 1) # [N], idx, masked ,0/1

    if adj_mat.is_sparse:
        idx1_mask = torch.sparse.sum(torch.index_select(adj_mat, 1, sorted_idx), dim = 1).to_dense() # [N]
    else:
        idx1_mask = adj_mat[: , sorted_idx].sum(dim = 1) # [N], idxs' in-neighbors, masked, weighted
    idx1_mask = idx_mask + idx1_mask # add selfloop
    idx1_mask[idx1_mask > 0] = 1 # [N], idx + their neighbors, masked ,0/1, float
    
    idx1 = torch.nonzero(idx1_mask).squeeze(1).to(on_device) # [?], remove 0s, idx + their neighbors, nodeidx
    idx1_to_adjsub_i = dict([(idx_, i) for i, idx_ in enumerate(idx1.tolist())])
    idx_in_adjsub = torch.tensor([idx1_to_adjsub_i[idx_] for idx_ in idx.tolist()], dtype = torch.long, device = on_device)

    adj_sub = torch.index_select(adj_mat, 0, idx1)
    adj_sub = torch.index_select(adj_sub, 1, idx1)
    if adj_mat.is_sparse:
        adj_sub = adj_sub.to_dense()

    if on_device != out_device:
        adj_sub = adj_sub.to(out_device)
        idx1 = idx1.to(out_device)
        idx_in_adjsub = idx_in_adjsub.to(out_device)

    return adj_sub, idx1, idx_in_adjsub 
