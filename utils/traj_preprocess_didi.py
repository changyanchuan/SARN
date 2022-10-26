
import pandas as pd
from ast import literal_eval
import logging
import random
from st_mapmatching import MapMatching

# (head -1 train.csv && (tail -n +2 train.csv | shuf -n 100000)) > tmp

osm_file_path = '../data/OSM_Chengdu_2thRings_raw'

wgs_file_path = '../data/didi_raw_150k_wgs'
df_file_path = '../data/didi_raw_150k'
sampled_file_path = '../data/didi_preprocess_len10150'
sampled_mm_file_path = '../data/didi_preprocess_len10150_mm'
mm_len_file_path = '../data/didi_len2200_mm'

len_range = (10, 150) # (10, 50)
lonlat_range = [104.0143000, 104.1176000, 30.6193000, 30.7034000] # 2th ring CD
mm_len_range = (2, 200) # (30, 60)
mm_output_size = 11000


# convert wgs raw file to df
# fix location shifts
# down sampling
def convert_raw_to_df(in_file, out_file):
    logging.info("convert_raw_to_pd starts.")

    lon_shift = -0.00248
    lat_shift = 0.00245
    downsample_rate = 0.2
    with open(in_file, 'r') as fh:
        trajs = []
        for line in fh:
            line = line.split(' ')
            traj_id = line[1]
            traj = []
            for i in range(3, len(line), 3 * int(1 / downsample_rate) ):
                traj.append([int(line[i]), float(line[i + 1]) + lon_shift, float(line[i + 2]) + lat_shift])
            trajs.append([traj_id, traj])
        
    df_trajs = pd.DataFrame(trajs, columns = ['TRIP_ID', 'POLYLINE']).set_index('TRIP_ID')
    
    df_trajs.to_csv(out_file)

    logging.info("convert_raw_to_pd end.")
    return


def sample_from_raw_data(in_file, out_file):
    logging.info("sample_from_raw_data starts.")

    pd_trajs = pd.read_csv(in_file, delimiter = ',', index_col = 'TRIP_ID')
    pd_trajs.loc[:,'POLYLINE'] = pd_trajs.POLYLINE.apply(literal_eval)

    pd_trajs['size'] = pd_trajs.POLYLINE.apply(lambda traj: len(traj))
    pd_trajs = pd_trajs[(pd_trajs['size'] >= len_range[0]) & \
                        (pd_trajs['size'] < len_range[1])]

    # output range
    pd_trajs['lon_min'] = pd_trajs.POLYLINE.apply(lambda traj: min(map(lambda p: p[1], traj)))
    pd_trajs['lon_max'] = pd_trajs.POLYLINE.apply(lambda traj: max(map(lambda p: p[1], traj)))
    pd_trajs['lat_min'] = pd_trajs.POLYLINE.apply(lambda traj: min(map(lambda p: p[2], traj)))
    pd_trajs['lat_max'] = pd_trajs.POLYLINE.apply(lambda traj: max(map(lambda p: p[2], traj)))
    
    logging.info("Raw traj range. lon=[{}, {}], lat=[{}, {}]".format( \
                min(pd_trajs.lon_min), max(pd_trajs.lon_max),
                min(pd_trajs.lat_min), max(pd_trajs.lat_max)))

    pd_trajs = pd_trajs[(pd_trajs['lon_min'] > lonlat_range[0]) & \
                        (pd_trajs['lon_max'] < lonlat_range[1]) & \
                        (pd_trajs['lat_min'] > lonlat_range[2]) & \
                        (pd_trajs['lat_max'] < lonlat_range[3])]
           
    pd_trajs.to_csv(out_file)

    logging.info("to_csv.shape={}".format(pd_trajs.shape))
    logging.info("sample_from_raw_data ends.")
    return 


def mapmatching(osm_file_prefix, traj_file_in, traj_file_out):

    mm = MapMatching(osm_file_prefix)
    lst_trajs = mm.read_trajs(traj_file_in)
    lst_trajs = lst_trajs[:mm_output_size] if mm_output_size > 0 else lst_trajs
    lst_trajs_mm = mm.traj_mapmatching_parallelwrapper(lst_trajs, num_cores = 6)
    # lst_trajs_mm = mm.traj_mapmatching(lst_trajs)
    df = pd.DataFrame(lst_trajs_mm, columns = ['traj_id', 'mm_edges', 'mm_coors', 'len_mm_edges', 'mm_timestamp', 'mm_center']).set_index('traj_id')
    df.to_csv(traj_file_out)
    logging.info("mapmatching ends.")
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
            return pd.Series([row['mm_edges'][start_i : start_i + mm_len_range[1]], \
                    row['mm_timestamp'][start_i : start_i + mm_len_range[1]], \
                    row['mm_center'][start_i : start_i + mm_len_range[1]]])


    pd_trajs['start_i'] =  pd_trajs.len_mm_edges.apply(random_trunc_idx)
    pd_trajs[['mm_edges', 'mm_timestamp', 'mm_center']] = pd_trajs.apply(trunc_by_idx, axis = 1)
    pd_trajs['len_mm_edges'] = pd_trajs.len_mm_edges.apply(lambda l: min(l, mm_len_range[1]))

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

    convert_raw_to_df(wgs_file_path, df_file_path)
    sample_from_raw_data(df_file_path, sampled_file_path)
    mapmatching(osm_file_path, sampled_file_path, sampled_mm_file_path)
    process_mmed_traj(sampled_mm_file_path, mm_len_file_path)
    pass