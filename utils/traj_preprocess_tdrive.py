import logging
import random
import pandas as pd
from ast import literal_eval
from st_mapmatching import MapMatching


osm_file_path = '../data/OSM_Beijing_2thRings_raw'

raw_file_path = '../data/tdrive_raw_17m'
sampled_file_path = '../data/tdrive_preprocess_len10150'
sampled_mm_file_path = '../data/tdrive_preprocess_len10150_mm'
mm_len_file_path = '../data/tdrive_len30120_mm'

len_range = (10, 150) #(20, 40)
max_traj_ts_gap = 20*60
lonlat_range = [116.3401000, 116.4425000, 39.8650000, 39.9503000] # 2th ring Beijing
mm_len_range = (30, 120) #(30, 60)
mm_output_size = 11000


# convert raw tdrive to df
# in raw tdrive file, each line is one gps record, and each driver has one trajectory.
# we read gps records and slice them to trajectories, and check their lonlat ranges
# maximum traj length and maximum duration gap are defined.
def convert_raw_to_df(in_file, out_file):
    logging.info("convert_raw_to_pd starts.")

    df = pd.read_csv(in_file, delimiter = ',', header = None, names=['driver_id', 'datetime', 'lon', 'lat'])
    df['ts'] = pd.to_datetime(df.datetime).astype('int64') // 10**9
    df['xyt'] = df[['ts','lon','lat']].apply(tuple, axis = 1)
    df2 = df.groupby('driver_id')['xyt'].apply(list).to_frame('traj')
    trajs = df2.to_dict(orient = 'index') #{driver_id: {traj:[...]}, }

    traj_new_id = 0
    trajs_new = []

    for driver_id, dic in trajs.items():
        traj_raw = dic['traj'] # list of 3-tuples

        # remove points that are outside the map lonlat range
        for i in range(len(traj_raw) - 1, -1, -1):
            if traj_raw[i][1] <= lonlat_range[0] \
                    or traj_raw[i][1] >= lonlat_range[1] \
                    or traj_raw[i][2] <= lonlat_range[2] \
                    or traj_raw[i][2] >= lonlat_range[3]:
                del traj_raw[i]

        # remove duplicates
        for i in range(len(traj_raw) - 1, 0, -1):
            if traj_raw[i][1] == traj_raw[i - 1][1] \
                    and traj_raw[i][2] == traj_raw[i - 1][2] \
                    or traj_raw[i][0] == traj_raw[i - 1][0]:
                del traj_raw[i]

        # slicing 
        if len(traj_raw) < len_range[0]:
            continue

        start_i = 0
        last_ts = traj_raw[0][0]
        check_traj = False
        
        for i in range(1, len(traj_raw)):
            ts, lon, lat = traj_raw[i]
            if ts - last_ts > max_traj_ts_gap:
                check_traj = True
            else:
                last_ts = ts

            if i - start_i >= len_range[1]:
                check_traj = True
            
            if i == len(traj_raw) - 1:
                check_traj = True

            if check_traj:
                if i - start_i >= len_range[0]:
                    trajs_new.append([traj_new_id, driver_id, traj_raw[start_i : i]])
                    traj_new_id += 1
                start_i = i
                check_traj = False
                last_ts = ts


    logging.info('#new_trajs={}'.format(traj_new_id))

    df_trajs_new = pd.DataFrame(trajs_new, columns=['TRIP_ID', 'driver_id', 'POLYLINE']).set_index('TRIP_ID')

    df_trajs_new.to_csv(out_file)

    logging.info("convert_raw_to_pd end.")
    return


def mapmatching(osm_file_prefix, traj_file_in, traj_file_out):

    mm = MapMatching(osm_file_prefix)
    lst_trajs = mm.read_trajs(traj_file_in)
    lst_trajs = lst_trajs[:mm_output_size] if mm_output_size > 0 else lst_trajs
    lst_trajs_mm = mm.traj_mapmatching_parallelwrapper(lst_trajs, num_cores = 6)
    # lst_trajs_mm = mm.traj_mapmatching(lst_trajs)
    df = pd.DataFrame(lst_trajs_mm, columns = ['traj_id', 'mm_edges', 'mm_coors', 'len_mm_edges', 'mm_timestamp', 'mm_center']).set_index('traj_id')
    df.to_csv(traj_file_out)
    logging.info("mapmatching end.")
    return


# truncate long mm traj
def process_mmed_traj(in_file, out_file):
    logging.info("process_mmed_traj starts.")

    pd_trajs = pd.read_csv(in_file, delimiter = ',', index_col = 'traj_id')
    pd_trajs.loc[:,'mm_edges'] = pd_trajs.mm_edges.apply(literal_eval)
    pd_trajs.loc[:,'mm_timestamp'] = pd_trajs.mm_timestamp.apply(literal_eval)
    pd_trajs.loc[:,'mm_center'] = pd_trajs.mm_center.apply(literal_eval)

    def random_trunc_idx(l):
        if l > mm_len_range[1]:
            start_i = random.randint(0, l-mm_len_range[1]-1)
            return start_i
        else:
            return -1

    def trunc_by_idx(row):
        if row['start_i'] < 0:
            return pd.Series([row['mm_edges'], row['mm_timestamp'], row['mm_center']])
        else:
            start_i = row['start_i']
            traj_len = mm_len_range[1]
            return pd.Series([row['mm_edges'][start_i : start_i + traj_len], \
                    row['mm_timestamp'][start_i : start_i + traj_len], \
                    row['mm_center'][start_i : start_i + traj_len]])


    pd_trajs['start_i'] =  pd_trajs.len_mm_edges.apply(random_trunc_idx)
    pd_trajs[['mm_edges', 'mm_timestamp', 'mm_center']] = pd_trajs.apply(trunc_by_idx, axis = 1)
    # pd_trajs['len_mm_edges'] = pd_trajs.len_mm_edges.apply(lambda l: min(l, mm_len_range[1]))
    pd_trajs['len_mm_edges'] = pd_trajs.mm_edges.apply(lambda l: len(l))

    pd_trajs = pd_trajs[(pd_trajs['len_mm_edges'] >= mm_len_range[0]) & \
                        (pd_trajs['len_mm_edges'] <= mm_len_range[1])]

    pd_trajs['mm_edges_hash'] = pd_trajs.mm_edges.apply(str)
    pd_trajs = pd_trajs.drop_duplicates(subset = 'mm_edges_hash')
    
    pd_trajs = pd_trajs[['mm_edges', 'len_mm_edges', 'mm_timestamp', 'mm_center']]
    pd_trajs.to_csv(out_file)

    logging.info("to_csv.shape={}".format(pd_trajs.shape))
    logging.info("process_mmed_traj ends.")
    return 



if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s")

    convert_raw_to_df(raw_file_path, sampled_file_path)
    mapmatching(osm_file_path, sampled_file_path, sampled_mm_file_path)
    process_mmed_traj(sampled_mm_file_path, mm_len_file_path)
    pass