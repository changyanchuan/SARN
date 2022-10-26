# create ground truth trajectory similarity dat

import os
import logging
import random
import time
import pickle
from itertools import combinations
import multiprocessing as mp

from config import Config as Config
from utils import tool_funcs
from utils.osm_loader import OSMLoader
from utils.traj_distance import frechet_dist_linear as frechet_dist


# async operator
def simi_comp_operator(datasets_coors, sub_idx_pairs):
    simi = []
    for (idx1, idx2) in sub_idx_pairs:
        # simi.append(tdist.hausdorff(np.array(datasets_coors[idx1]), np.array(datasets_coors[idx2])))
        simi.append(frechet_dist(datasets_coors[idx1], datasets_coors[idx2]))
    logging.debug('simi_comp_operator ends. pid={}'.format(os.getpid()))
    return simi


def traj_simi_computation(osm_data: OSMLoader, dic_trajs: dict):
    # dic_trajs: {index -> {column -> value}}
    # 1. verify traj segments exist in osm_data.segid_in_adj_segments_graph, remove illegal trajs
    # 2. shuffle trajs and split to 3 datasets
    # 3. calculate simi in 3 datasets separately.
    # 4. dump 3 datasets

    _time = time.time()
    logging.info("traj_simi_computation starts at {}".format(_time))

    seg_ids = osm_data.segid_in_adj_segments_graph # list
    seg_id_to_idx = osm_data.seg_id_to_idx_in_adj_seg_graph
    lst_trajs = [] # all legal trajs [(trajid, traj), (), ...]

    # 1.
    for traj_id, dic in dic_trajs.items():
        traj = dic['mm_edges']
        _is_traj_legal = True
        for seg in traj:
            if seg not in seg_id_to_idx:
                _is_traj_legal = False
                break
        if _is_traj_legal:
            lst_trajs.append((traj_id, traj))
    assert len(lst_trajs) >= 10000
    lst_trajs = lst_trajs[:10000]
    logging.info("traj dataset sizes. #inputs={}. #legal={}" \
                    .format(len(dic_trajs), len(lst_trajs)))

    # 2.
    random.shuffle(lst_trajs)
    _len = len(lst_trajs)

    trains = lst_trajs[0 : int(_len * 0.8)] # [ (traj_id, [segid, segid, ...]), ...]
    evals = lst_trajs[int(_len * 0.8) : int(_len * 0.9)]
    tests = lst_trajs[int(_len * 0.9) : ]
    logging.info("traj dataset sizes. #inputs={}. Legal traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(len(dic_trajs), _len, len(trains), len(evals), len(tests)))
    
    # 3.
    df_segs = osm_data.segments[['inc_id','s_lon','s_lat','e_lon','e_lat']].reset_index().set_index('inc_id')
    tests_simi = simi_matrix(tests, df_segs)
    evals_simi = simi_matrix(evals, df_segs)
    trains_simi = simi_matrix(trains, df_segs) # [ [simi, simi, ... ], ... ]

    max_distance = max( max(map(max, trains_simi)), max(map(max, evals_simi)), max(map(max, tests_simi)) )

    # 4.
    dic = {'trains': trains, 'evals': evals, 'tests': tests, \
            'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
            'max_distance': max_distance}
    _dic_file = '{}/data/{}_traj_simi_dict.pickle'.format(Config.root_dir, Config.trajsimi_prefix)
    with open(_dic_file, 'wb') as fh:
        pickle.dump(dic, fh, protocol = pickle.HIGHEST_PROTOCOL)
    
    logging.info("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return dic


def simi_matrix(datasets, df_segs):
    # datasets = [ (traj_id, [segid, segid, ...]), (), ... ]
    __t = time.time()
    datasets_coors = []
    datasets_simi = []
    datasets_idx_pairs = list(combinations(range(len(datasets)), 2)) # [(0,1), (0,2), ...]

    # convert segid to lonlat
    for i, (traj_id, traj) in enumerate(datasets):
        coors = []
        segid = traj[0]
        _seg_df = df_segs.loc[segid, ['s_lon','s_lat']]
        coors.append([_seg_df['s_lon'], _seg_df['s_lat']])

        for segid in traj:
            _seg_df = df_segs.loc[segid, ['e_lon','e_lat']]
            coors.append([_seg_df['e_lon'], _seg_df['e_lat']])
        datasets_coors.append(coors)
    
    # compute similarity of traj pairs in parallel
    num_cores = 8
    datasets_slice_idxs = tool_funcs.slicing(len(datasets_idx_pairs), num_cores)
    logging.debug('simi matrix computation starts. parent_pid={}, #processors={}, slices={}' \
                    .format(os.getpid(), len(datasets_slice_idxs), datasets_slice_idxs))
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(simi_comp_operator, (datasets_coors, datasets_idx_pairs[idx[0]: idx[1]], )) for idx in datasets_slice_idxs]
    pool.close()
    pool.join()

    simis = []
    for r in results: # timeout not allowed here
        if type(r.get()) == list:
            simis.extend(r.get()) #[x, x, x, ...] 1d
        else:
            logging.info('simi matrix computation error. return type={}'.format(type(r.get())))
            exit(-9999)
        
    # extend simis to a simi matrix, and pad 0s.
    _simis_idx = 0
    for i in range(len(datasets)):
        _simis_idx_end = _simis_idx + len(datasets) - i - 1
        datasets_simi.append( [0]*(i+1) + simis[_simis_idx : _simis_idx_end])
        _simis_idx = _simis_idx_end
    assert _simis_idx == len(simis)

    logging.debug('simi_matrix computation done. #datasets={}, @={:.3f}'.format(len(datasets), time.time()- __t))
    return datasets_simi

