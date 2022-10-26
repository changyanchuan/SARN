# compute ground truth shortest-path distances
# based on road network graphs

import os
import logging
import multiprocessing as mp
import networkx as nx

from utils import tool_funcs
from utils.osm_loader import OSMLoader


# if the number of segments is large, 
# we need to save each result slice into file,
# then merge them.
_g_adj = None # for simplicity and efficiency
_segid_to_idx = None # it's a list, not a dict!! for efficiency
def spd_comp_operator(sourcenodes):
    global _g_adj, _segid_to_idx
    dist = []
    for nodeid in sourcenodes:
        spds = nx.shortest_path_length(_g_adj, source = nodeid, weight = 's1_length') # dict
        lst = [-1] * _g_adj.number_of_nodes()
        for k, v in spds.items():
            # use int here. otherwise, the dumped spd file will be huge!
            lst[_segid_to_idx[k]] = int(v) 
        dist.append(lst)  
    logging.debug('spd_comp_operator ends. pid={}'.format(os.getpid()))
    return dist


def calculate_spd_dict(osm: OSMLoader, num_cores = 28):
    global _g_adj, _segid_to_idx
    _g_adj = osm.adj_segments_graph
    
    max_id_ =  max([id_ for id_, idx_ in osm.seg_id_to_idx_in_adj_seg_graph.items()])
    _segid_to_idx = [-1] * (max_id_ + 1)
    for id_, idx_ in osm.seg_id_to_idx_in_adj_seg_graph.items():
        _segid_to_idx[id_] = idx_
    
    nodes = osm.segid_in_adj_segments_graph
    datasets_slice_idxs = tool_funcs.slicing(len(nodes), num_cores*10)
    logging.debug('spd matrix computation starts. parent_pid={}, #processors={}, slices={}' \
                    .format(os.getpid(), len(datasets_slice_idxs), datasets_slice_idxs))
    
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(spd_comp_operator, (nodes[idx[0]: idx[1]], )) for idx in datasets_slice_idxs]
    pool.close()
    pool.join()

    lst_spd = []
    for r in results: # timeout not allowed here
        if type(r.get()) == list:
            lst_spd.extend(r.get())
        else:
            logging.info('spd matrix computation error. return type={}'.format(type(r.get())))
            exit(-9999)
    return lst_spd
