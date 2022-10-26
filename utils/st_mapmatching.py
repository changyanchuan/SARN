# https://github.com/rottenivy

import time
import numpy as np
import pandas as pd
import networkx as nx
import warnings
import logging
import math
from ast import literal_eval
import multiprocessing as mp
import os

from tool_funcs import haversine_np as haversine
from tool_funcs import haversine as haversine_py
from tool_funcs import pairwise, slicing

pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')
warnings.filterwarnings('ignore')

class MapMatching:
    def __init__(self, file_prefix):
        self.vertice_file = file_prefix + '_node'
        self.segment_file = file_prefix + '_segment'

        if 'Chengdu' in file_prefix:
            lon_100m = 0.0010507143 # CD
            lat_100m = 0.0008982567 # CD
        elif 'Beijing' in file_prefix:
            lon_100m = 0.001172332943 # Beijing
            lat_100m = 0.000899280575 # Beijing
        elif 'SanFrancisco' in file_prefix:
            lon_100m = 0.001137656428 # sf
            lat_100m = 0.0008992805755 # sf


        self.loninter = lon_100m
        self.latinter = lat_100m

        self.observation_std = 20
        self.temporal_speed_limit = 20

        self.nx_vertice, self.nx_edge, self.vertice_dict, self.edge_dict, self.roadnetwork \
                = self._init_network_data()


    def _init_network_data(self):

        nx_vertice = pd.read_csv(self.vertice_file, usecols=['node_id', 'lon', 'lat'])
        vertice_dict = nx_vertice.set_index('node_id').T.to_dict('list')

        nx_edge = pd.read_csv(self.segment_file, usecols=['inc_id', 's_id', 'e_id', 's_lon', 's_lat', 'e_lon', 'e_lat', 'c_lon', 'c_lat', 'length'])
        nx_edge['mbr_lon_lo'] = nx_edge[['s_lon', 'e_lon']].min(axis=1)
        nx_edge['mbr_lon_hi'] = nx_edge[['s_lon', 'e_lon']].max(axis=1)
        nx_edge['mbr_lat_lo'] = nx_edge[['s_lat', 'e_lat']].min(axis=1)
        nx_edge['mbr_lat_hi'] = nx_edge[['s_lat', 'e_lat']].max(axis=1)
        nx_edge['distance'] = nx_edge['length']

        edge_dict = nx_edge.set_index('inc_id').T.to_dict('list')

        roadnetwork = nx.from_pandas_edgelist(nx_edge,
                    's_id', 'e_id', edge_attr = True, create_using = nx.DiGraph())
        
        logging.info("_init_network_data ends.")
        return nx_vertice, nx_edge, vertice_dict, edge_dict, roadnetwork


    def read_trajs(self, in_file):

        def lonlatlist_to_dfTLONLAT(lst):
            df = pd.DataFrame(lst, columns = ['T', 'LON', 'LAT'])
            # df['T'] = df.index
            # df = df[['T', 'LON', 'LAT']]
            return df

        pd_trajs = pd.read_csv(in_file, delimiter = ',', index_col = 'TRIP_ID')
        pd_trajs.loc[:,'POLYLINE'] = pd_trajs.POLYLINE.apply(literal_eval)
        dic_trajs = pd_trajs[['POLYLINE']].to_dict('dict')
        lst_trajs = list(map(lambda x: (x[0], lonlatlist_to_dfTLONLAT(x[1])), dic_trajs['POLYLINE'].items()))

        return lst_trajs # [(id, dataframe), ... ] # dataframe = [LON, LAT, T]


    def traj_mapmatching(self, lst_trajs: list):
        _t = time.time()
        _t10 = time.time()
        lst_trajs_mm = []
        for i, (traj_id, traj) in enumerate(lst_trajs): # traj: df
            try:
                pd_mm_edge = self._traj_mapmatching_core(traj_id, traj)

                logging.debug('{} - MATCHED_EDGE==='.format(i))
                logging.debug([(i, v) for i, v in enumerate(pd_mm_edge.iloc[0].loc['MATCHED_EDGE'])])
                logging.debug('{} - MATCHED_NODE==='.format(i))
                logging.debug([(i, v) for i, v in enumerate(pd_mm_edge.iloc[0].loc['MATCHED_NODE'])])
                
                mm_edge, mm_coor = self._connect_edge_seq(traj, pd_mm_edge)
                ts, centers = self.__fill_timestamp_and_center(traj, mm_edge)
                lst_trajs_mm.append([traj_id, mm_edge, mm_coor, len(mm_edge), ts, centers])
            except:
                continue
            finally:
                if (i) % 100 == 0:
                    logging.info("finished {} trajs. pid={}, @={:.3f}".format(i, os.getpid(), time.time() - _t10))
                    _t10 = time.time()
            
        logging.info("[single] @={:.3f}, #input_trajs={}, #out_trajs={}".format(time.time() - _t, len(lst_trajs), len(lst_trajs_mm)))  
        return lst_trajs_mm

    def traj_mapmatching_parallelwrapper(self, lst_trajs, num_cores = 1):
        if num_cores <= 1 or len(lst_trajs) < num_cores:
            return self.traj_mapmatching(lst_trajs)

        logging.info("multiprocessing start. parent_pid={}".format(os.getpid()))
        _t = time.time()
        
        slice_range_idxs = slicing(len(lst_trajs), num_cores)
        logging.info("multiprocessing. slice={}".format(slice_range_idxs))

        pool = mp.Pool(num_cores)
        # results = pool.map(self.traj_mapmatching, [lst_trajs[idx[0]: idx[1]] for idx in slice_range_idxs ])
        results = [pool.apply_async(self.traj_mapmatching, (lst_trajs[idx[0]: idx[1]], )) for idx in slice_range_idxs]

        # results = []
        # for idx in slice_range_idxs:
        #     results.append(pool.apply_async(self.traj_mapmatching, (lst_trajs[idx[0]: idx[1]], )))

        pool.close()
        pool.join()

        lst_trajs_mm = []
        for r in results:
            if type(r.get()) == list:
                lst_trajs_mm.extend(r.get())
            else:
                logging.info('traj mm parallel single processor error. return type={}'.format(type(r.get())))
                exit(-9999)
            
        logging.info("[parallel summary] @={:.3f}, #input_trajs={}, #out_trajs={}".format(time.time() - _t, len(lst_trajs), len(lst_trajs_mm)))  
        return lst_trajs_mm


    def _traj_mapmatching_core(self, traj_id, traj):
        # traj = [ 'T', 'LON, 'LAT',
        # 'CAND_ND_DIS', 'CAND_EG', 'CAND_ND', 
        # 'N', 'V', 'F', 'pre']

        traj.drop_duplicates(['LON', 'LAT'], inplace = True)
        cand_results = traj.apply(self._get_candidates, axis = 1)

        traj['CAND_ND_DIS'] = [x[2] if x != -1 else -1 for x in cand_results]
        traj['CAND_EG'] = [x[0] if x != -1 else -1 for x in cand_results]
        traj['CAND_ND'] = [x[1] if x != -1 else -1 for x in cand_results]
        traj = traj[traj['CAND_EG'] != -1]

        if traj.shape[0] > 1:  # not enough candidates
            traj['N'] = traj.apply(self._observation_probability, axis=1)
            traj['V'] = self._transmission_probability(traj)
            # traj['F']= self._spatial_analysis(traj)

            # traj['TT'] = self._temporal_probability(traj)
            traj['F'], traj['pre'] = self._spatial_analysis2(traj, False)
            
            matched_edge_list = []
            matched_node_list = []
            i_eg = traj['F'].iloc[-1].index(max(traj['F'].iloc[-1]))
            matched_edge_list.append(traj['CAND_EG'].iloc[-1][i_eg])
            matched_node_list.append(traj['CAND_ND'].iloc[-1][i_eg])
            i_eg = traj['pre'].iloc[-1][i_eg]

            
            for i in range(traj.values.shape[0]-2, -1, -1):
                matched_edge_list.append(traj['CAND_EG'].iloc[i][i_eg])
                matched_node_list.append(traj['CAND_ND'].iloc[i][i_eg])
                i_eg = traj['pre'].iloc[i][i_eg]
            matched_edge_list.reverse()
            matched_node_list.reverse()
            return pd.DataFrame([[traj_id, matched_edge_list, matched_node_list]], columns=['TRAJ_ID', 'MATCHED_EDGE', 'MATCHED_NODE'])
        else:
            return pd.DataFrame([[traj_id, -1, -1]], columns=['TRAJ_ID', 'MATCHED_EDGE', 'MATCHED_NODE'])


    def _get_candidates(self, row):
        traj_point = [row['LON'], row['LAT']]
        
        sub_nx_edge = self.nx_edge[( \
                (self.nx_edge['mbr_lon_lo']-self.loninter <= traj_point[0]) \
                & (self.nx_edge['mbr_lon_hi']+self.loninter >= traj_point[0]) \
                & (self.nx_edge['mbr_lat_lo']-self.latinter <= traj_point[1]) \
                & (self.nx_edge['mbr_lat_hi']+self.latinter >= traj_point[1]))]
        cand_edges = self._get_traj2edge_distance(traj_point, sub_nx_edge)
        cand_edges = cand_edges[(cand_edges['shortest_dist'] <= 35) & pd.notnull(cand_edges['shortest_dist'])]
        cand_edges['shortest_dist'] = round(cand_edges['shortest_dist'])
        if not cand_edges.empty:
            logging.debug(str(row.index) + '--' + str(cand_edges['inc_id'].tolist()))
            return cand_edges['inc_id'].tolist(), cand_edges['matched_nd'].tolist(), cand_edges['shortest_dist'].tolist()
        else:
            return -1, -1, -1


    def _get_traj2edge_distance(self, traj_point, sub_nx_edge):
        sub_nx_edge['a'] = haversine(traj_point[0], traj_point[1], sub_nx_edge['s_lon'], sub_nx_edge['s_lat'])
        sub_nx_edge['b'] = haversine(traj_point[0], traj_point[1], sub_nx_edge['e_lon'], sub_nx_edge['e_lat'])
        sub_nx_edge['c'] = haversine(sub_nx_edge['s_lon'], sub_nx_edge['s_lat'], sub_nx_edge['e_lon'], sub_nx_edge['e_lat'])
        indexer1 = sub_nx_edge['b']**2 > sub_nx_edge['a']**2 + sub_nx_edge['c']**2
        indexer2 = sub_nx_edge['a']**2 > sub_nx_edge['b']**2 + sub_nx_edge['c']**2
        sub_nx_edge.loc[indexer1, 'shortest_dist'] = sub_nx_edge.loc[indexer1, 'a']
        sub_nx_edge.loc[indexer1, 'matched_nd'] = sub_nx_edge.loc[indexer1, 's_id']
        sub_nx_edge.loc[indexer2, 'shortest_dist'] = sub_nx_edge.loc[indexer2, 'b']
        sub_nx_edge.loc[indexer2, 'matched_nd'] = sub_nx_edge.loc[indexer2, 'e_id']

        sub_nx_edge['l'] = (sub_nx_edge['a'] + sub_nx_edge['b'] + sub_nx_edge['c'])/2
        sub_nx_edge['s'] = np.sqrt(sub_nx_edge['l'] * np.abs(sub_nx_edge['l'] - sub_nx_edge['a']) * np.abs(sub_nx_edge['l'] - sub_nx_edge['b']) * np.abs(sub_nx_edge['l'] - sub_nx_edge['c']))

        indexer3 = pd.isnull(sub_nx_edge['shortest_dist'])
        sub_nx_edge.loc[indexer3, 'shortest_dist'] = 2 * sub_nx_edge.loc[indexer3, 's'] / sub_nx_edge.loc[indexer3, 'c']

        return sub_nx_edge[['inc_id', 'shortest_dist', 'matched_nd']]


    def _observation_probability(self, row):
        cand_nd_df = np.array(row['CAND_ND_DIS'])
        cand_nd_df = 1 / (np.sqrt(2 * np.pi) * self.observation_std) * np.exp(-(cand_nd_df ** 2) / (2*self.observation_std*self.observation_std))
        # cand_nd_df[cand_nd_df >= 0] = 1
        return list(cand_nd_df)


    def _transmission_probability(self, traj):
        # beta = 50
        v_list = [[]]
        for row1, row2 in pairwise(traj.values):
            d = haversine(row1[1], row1[2], row2[1], row2[2])
            d_normlized = d
            row_v_list = []
            for idx2, nd2 in enumerate(row2[-2]):
                temp_list = []
                for idx1, nd1 in enumerate(row1[-2]):

                    try:  # nd1 and nd2 are not connected
                        if pd.notnull(nd1) and pd.notnull(nd2):
                            temp_list.append(d / (d_normlized+nx.astar_path_length(self.roadnetwork, nd1, nd2, weight='distance')))
                            # temp_list.append(1 / beta * np.exp(- np.abs(d - nx.astar_path_length(self.roadnetwork, nd1, nd2, weight='distance'))/ beta))
                        elif pd.notnull(nd1):
                            nd2_forward_node = self.edge_dict[row2[-3][idx2]][1]
                            nd2_forward_node_cor = self.vertice_dict[nd2_forward_node]
                            dist = ( \
                                    nx.astar_path_length(self.roadnetwork, nd1, nd2_forward_node, weight='distance') \
                                    - np.sqrt(np.abs(haversine(row2[1], row2[2], nd2_forward_node_cor[0], nd2_forward_node_cor[1])**2 - row2[-4][idx2]**2)) \
                                    )
                            if dist > 0:
                                temp_list.append(d / (d_normlized+dist))
                                # temp_list.append(1 / beta * np.exp(- np.abs(d - dist)/ beta))
                            else:
                                nd2_back_node = self.edge_dict[row2[-3][idx2]][0]
                                nd2_back_node_cor = self.vertice_dict[nd2_back_node]
                                temp_list.append(d / (d_normlized+nx.astar_path_length(self.roadnetwork, nd1, nd2_back_node, weight='distance') + np.sqrt(np.abs(haversine(row2[1], row2[2], nd2_back_node_cor[0], nd2_back_node_cor[1])**2 - row2[-4][idx2]**2))))
                                # temp_list.append(1 / beta * np.exp(- np.abs(d - (nx.astar_path_length(self.roadnetwork, nd1, nd2_back_node, weight='distance') + np.sqrt(np.abs(haversine(row2[1], row2[2], nd2_back_node_cor[0], nd2_back_node_cor[1])**2 - row2[-4][idx2]**2))))/ beta))
                        elif pd.notnull(nd2):
                            nd1_back_node = self.edge_dict[row1[-3][idx1]][0]
                            nd1_back_node_cor = self.vertice_dict[nd1_back_node]
                            dist = ( \
                                    nx.astar_path_length(self.roadnetwork, nd1_back_node, nd2, weight='distance') \
                                    - np.sqrt(np.abs(haversine(row1[1], row1[2], nd1_back_node_cor[0], nd1_back_node_cor[1]) ** 2 - row1[-4][idx1] ** 2)) \
                                    )
                            if dist > 0:
                                temp_list.append(d / (d_normlized+dist))
                                # temp_list.append(1 / beta * np.exp(- np.abs(d - dist)/ beta))
                            else:
                                nd1_forward_node = self.edge_dict[row1[-3][idx1]][1]
                                nd1_forward_node_cor = self.vertice_dict[nd1_forward_node]
                                temp_list.append(d / (d_normlized+nx.astar_path_length(self.roadnetwork, nd1_forward_node, nd2, weight='distance') + np.sqrt(np.abs(haversine(row1[1], row1[2], nd1_forward_node_cor[0], nd1_forward_node_cor[1]) ** 2 - row1[-4][idx1] ** 2))))
                                # temp_list.append(1 / beta * np.exp(- np.abs(d - (nx.astar_path_length(self.roadnetwork, nd1_forward_node, nd2, weight='distance') + np.sqrt(np.abs(haversine(row1[1], row1[2], nd1_forward_node_cor[0], nd1_forward_node_cor[1]) ** 2 - row1[-4][idx1] ** 2))))/ beta))
                        else:
                            nd1_back_node = self.edge_dict[row1[-3][idx1]][0]
                            nd1_back_node_cor = self.vertice_dict[nd1_back_node]
                            nd2_forward_node = self.edge_dict[row2[-3][idx2]][1]
                            nd2_forward_node_cor = self.vertice_dict[nd2_forward_node]
                            dist = ( \
                                    nx.astar_path_length(self.roadnetwork, nd1_back_node, nd2_forward_node, weight='distance') \
                                    - np.sqrt(np.abs(haversine(row1[1], row1[2], nd1_back_node_cor[0], nd1_back_node_cor[1]) ** 2 - row1[-4][idx1] ** 2)) \
                                    - np.sqrt(np.abs(haversine(row2[1], row2[2], nd2_forward_node_cor[0], nd2_forward_node_cor[1]) ** 2 - row2[-4][idx2] ** 2)))
                            if dist > 0:
                                temp_list.append(d / (d_normlized+dist))
                                # temp_list.append(1 / beta * np.exp(- np.abs(d - dist)/ beta))
                            else:
                                nd1_forward_node = self.edge_dict[row1[-3][idx1]][1]
                                nd1_forward_node_cor = self.vertice_dict[nd1_forward_node]
                                nd2_back_node = self.edge_dict[row2[-3][idx2]][0]
                                nd2_back_node_cor = self.vertice_dict[nd2_back_node]
                                temp_list.append(d / (d_normlized+nx.astar_path_length(self.roadnetwork, nd1_forward_node, nd2_back_node, weight='distance') + np.sqrt(np.abs(haversine(row1[1], row1[2], nd1_forward_node_cor[0], nd1_forward_node_cor[1]) ** 2 - row1[-4][idx1] ** 2)) + np.sqrt(np.abs(haversine(row2[1], row2[2], nd2_back_node_cor[0], nd2_back_node_cor[1]) ** 2 - row2[-4][idx2] ** 2))))
                                # temp_list.append(1 / beta * np.exp(- np.abs(d - (d_normlized+nx.astar_path_length(self.roadnetwork, nd1_forward_node, nd2_back_node, weight='distance') + np.sqrt(np.abs(haversine(row1[1], row1[2], nd1_forward_node_cor[0], nd1_forward_node_cor[1]) ** 2 - row1[-4][idx1] ** 2)) + np.sqrt(np.abs(haversine(row2[1], row2[2], nd2_back_node_cor[0], nd2_back_node_cor[1]) ** 2 - row2[-4][idx2] ** 2))))/ beta))
                    except:
                        temp_list.append(0)
                row_v_list.append(temp_list)
            v_list.append(row_v_list)
        return v_list


    def _spatial_analysis(self, row):
        return [[n_i * v_i[i] if not np.isinf(v_i[i]) else n_i for v_i in row[-1]] for i, n_i in enumerate(row[-2])]


    def _spatial_analysis2(self, traj, temporal_available):
        f_list = []
        pre_list = []

        f_list.append(traj['N'].iloc[0])
        pres = [i for i in range(len(traj['N'].iloc[0]))]
        pre_list.append(pres)


        for i_point in range(1, traj.values.shape[0]):
            fs = []
            pres = []
            for i_c in range(len(traj['V'].iloc[i_point])):
                # print(str(i_point) + ' ' +  str(i_c) + ' ' + str(len(traj['V'][i_point])))
                att_max = -1.0
                pre_i_1 = None

                for i_c_1 in range(len(traj['V'].iloc[i_point][i_c])):
                    f = (traj['TT'].iloc[i_point][i_c][i_c_1] if temporal_available else 1) \
                            * traj['V'].iloc[i_point][i_c][i_c_1] * traj['N'].iloc[i_point][i_c] if not np.isinf(traj['V'].iloc[i_point][i_c][i_c_1]) else traj['N'].iloc[i_point][i_c]
                    f += f_list[i_point-1][i_c_1]
                    if f > att_max:
                        att_max = f
                        pre_i_1 = i_c_1
                fs.append(att_max)
                pres.append(pre_i_1)
            f_list.append(fs)
            pre_list.append(pres)
        
        return f_list, pre_list


    def _temporal_probability(self, traj):
        t_list = [[]]
        for row1, row2 in pairwise(traj.values):
            d = haversine(row1[1], row1[2], row2[1], row2[2])
            d_normlized = d
            row_t_list = []
            for idx2, nd2 in enumerate(row2[-3]):
                temp_list = []
                for idx1, nd1 in enumerate(row1[-3]):

                    V = row2[-1][idx2][idx1] # V can be 0
                    try:
                        shortest_dist = d / V - d
                        speed = shortest_dist / (row2[0] - row1[0])
                        temp_list.append(  math.exp( -((max(speed, self.temporal_speed_limit) - self.temporal_speed_limit) / 5) )  )
                    except:
                        temp_list.append(0)
                row_t_list.append(temp_list)
            t_list.append(row_t_list)
        return t_list


    def _connect_edge_seq(self, traj, mm_pd):
        mm_edge = []
        mm_node = []
        mm_coor = []
        last_node = mm_pd.iloc[0].loc['MATCHED_NODE'][0] \
                if pd.notnull(mm_pd.iloc[0].loc['MATCHED_NODE'][0]) \
                else self.edge_dict[mm_pd.iloc[0].loc['MATCHED_EDGE'][0]][0]
        
        for i in range(1, len(mm_pd.iloc[0].loc['MATCHED_NODE'])):
            if pd.notnull(mm_pd.iloc[0].loc['MATCHED_NODE'][i]):
                if mm_pd.iloc[0].loc['MATCHED_NODE'][i] == last_node:
                    pass
                else:
                    s_path = nx.shortest_path(self.roadnetwork, last_node, mm_pd.iloc[0].loc['MATCHED_NODE'][i], weight='distance')
                    for start_nd, end_nd in pairwise(s_path):
                        mm_edge.append(self.roadnetwork.get_edge_data(start_nd, end_nd)['inc_id'])
                        mm_node.append(start_nd)
                        # mm_coor.append(1)
                        mm_coor.append(self.vertice_dict[start_nd][0])
                        mm_coor.append(self.vertice_dict[start_nd][1])

                    
                last_node = mm_pd.iloc[0].loc['MATCHED_NODE'][i]
            else:
                if self.edge_dict[mm_pd.iloc[0].loc['MATCHED_EDGE'][i]][1] == last_node:
                    pass
                else:
                    s_path = nx.shortest_path(self.roadnetwork, last_node, self.edge_dict[mm_pd.iloc[0].loc['MATCHED_EDGE'][i]][1], weight='distance')
                    for start_nd, end_nd in pairwise(s_path):
                        mm_edge.append(self.roadnetwork.get_edge_data(start_nd, end_nd)['inc_id'])
                        mm_node.append(start_nd)
                        # mm_coor.append(1)
                        mm_coor.append(self.vertice_dict[start_nd][0])
                        mm_coor.append(self.vertice_dict[start_nd][1])
                    
                last_node = self.edge_dict[mm_edge[-1]][1]

        mm_node.append(self.edge_dict[mm_edge[-1]][1])
        # mm_coor.append(1)
        mm_coor.append(self.vertice_dict[mm_node[-1]][0])
        mm_coor.append(self.vertice_dict[mm_node[-1]][1])
        
        # print(mm_edge)
        # print(*mm_coor, sep=' ')
        return mm_edge, mm_coor

    def __fill_timestamp_and_center(self, traj, mm_edge):
        avg_speed = 8 # m/s

        start_time = traj.iloc[0].loc['T']
        end_time = traj.iloc[-1].loc['T']

        centers = []
        for edge in mm_edge:
            centers.append([self.edge_dict[edge][6], self.edge_dict[edge][7]])

        timestamps = [start_time]
        for c1, c2 in pairwise(centers):
            dist = haversine_py(c1[0], c1[1], c2[0], c2[1])
            timestamps.append(timestamps[-1] + int(math.ceil(dist / avg_speed)))
        return timestamps, centers


def candidate_graph(traj):
    max_f = max([max([max(f) for f in f_list]) for f_list in traj['F'].tolist()[1:]])
    cand_graph = nx.DiGraph()
    idx = 0
    for row1, row2 in pairwise(traj.values):
        for i, nd2 in enumerate(row2[-5]):
            for j, nd1 in enumerate(row1[-5]):
                cand_graph.add_edge(str(idx) + '-' + str(nd1), str(idx + 1) + '-' + str(nd2), distance=max_f - row2[-1][i][j])
        idx += 1
    return cand_graph


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s")

    osm_file_prefix = '../data/OSM_Chengdu_2thRings_raw'
    traj_file = '../data/didi_preprocess_2k'
    traj_out_file = traj_file + '_mm'

    mm = MapMatching(osm_file_prefix)
    lst_trajs = mm.read_trajs(traj_file)
    lst_trajs_mm = mm.traj_mapmatching(lst_trajs)

    df = pd.DataFrame(lst_trajs_mm, columns = ['traj_id', 'mm_edges', 'mm_coors', 'len_mm_edges']).set_index('traj_id')
    df.to_csv(traj_out_file)

