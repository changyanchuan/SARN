# https://docs.python.org/3/library/xml.etree.elementtree.html#module-xml.etree.ElementTree

import sys
sys.path.append('..')

import pickle
import copy
import math
from itertools import tee
import numpy as np
np.seterr(divide = 'ignore')
import xml.etree.ElementTree as ET

from utils.cell import CellSpace


# ref: https://wiki.openstreetmap.org/wiki/Key:highway
highway_cls = {
    'motorway': 0,
    'motorway_link': 0,
    'trunk': 0,
    'trunk_link': 0,

    'primary': 1,
    'primary_link': 1,

    'secondary': 2,
    'secondary_link': 2,

    'tertiary': 3,
    'tertiary_link': 3,
    'unclassified': 3,

    'residential': 4,
    'living_street': 4,

    'service': 5,
    'road': 5
}

highway_cls_to_weight = {
    '0': 6.0,
    '1': 5.0,
    '2': 4.0,
    '3': 3.0,
    '4': 2.0,
    '5': 1.0,
}

rings_ranges = {
    'OSM_Chengdu_2thRings_raw': {'lat_lo': 30.6140780, 'lat_hi': 30.7051584, \
            'lon_lo': 104.0064460, 'lon_hi': 104.1241138},
    'OSM_Beijing_2thRings_raw': {'lat_lo': 39.8650000, 'lat_hi': 39.9503000, \
            'lon_lo': 116.3401000, 'lon_hi': 116.4425000},
    'OSM_SanFrancisco_downtown_raw': {'lat_lo': 37.7617000, 'lat_hi': 37.8131000, \
            'lon_lo': -122.4450000, 'lon_hi': -122.3803000},
}

lonlat_units = {
    'OSM_Chengdu_2thRings_raw': {'lon_unit': 0.010507143, 'lat_unit': 0.008982567},
    'OSM_Beijing_2thRings_raw': {'lon_unit': 0.01172332943, 'lat_unit': 0.00899280575},
    'OSM_SanFrancisco_downtown_raw': {'lon_unit': 0.00568828214, 'lat_unit': 0.004496402877}
}

lon_unit = 0.0
lat_unit = 0.0

dict_nodes = {} # id -> {lat, lon}
dict_ways = {} # 
dict_legal_nodes = {} # exclude the nodes whose highway types are not legal or those are not in the given area.

dict_segments = {}
dict_segments_way_ids = {}

dict_cells_by_index = {} # (lon_index, lat_index) -> [lon_lo, lat_lo, lon_hi, lat_hi]
dict_adj_cells = {} # (s_cell_id, e_cell_id) -> weight

ring_ranges = {}

def in_range(lon: float, lat: float):
    global ring_ranges
    if ring_ranges['lat_lo'] <= lat and lat <= ring_ranges['lat_hi'] and \
            ring_ranges['lon_lo'] <= lon and lon <= ring_ranges['lon_hi']:
        return True
    return False


def is_oneway(v):
    v = str(v)
    if v == 'yes' or v == '-1':
        return True
    return False


def get_highway_weight(highway_type: str):
    # highway_type = 'primary'
    return highway_cls_to_weight[str(highway_cls[highway_type])]


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return round(c * 6367 * 1000, 3)


def radian(lon1, lat1, lon2, lat2):
    dy = lat2 - lat1
    dx = lon2 - lon1
    r = 0.0
    if dx == 0:
        if dy >= 0:
            r = 1.5707963267948966 # math.pi / 2
        else: 
            r = 4.71238898038469 # math.pi * 1.5
        return round(r, 3)
    
    r = math.atan(dy / dx)
    # angle_in_degrees = math.degrees(angle_in_radians)
    if dx < 0:
        r = r + 3.141592653589793
    else:
        if dy < 0:
            r = r + 6.283185307179586
        else:
            pass
    return round(r, 3)



# read all legal nodes and ways from the input OSM file
def read_OSM_file(input_file_name):

    global dict_nodes
    global dict_ways

    tree = ET.parse(input_file_name)
    root = tree.getroot()

    _dict_nodes_legal = {}

    # all nodes in the small range
    for ele in root:
        if ele.tag == 'node' and 'id' in ele.attrib:
            if in_range(float(ele.get('lon')), float(ele.get('lat'))):
                dict_nodes[ele.get('id')] = {'lon': float(ele.get('lon')), 'lat': float(ele.get('lat'))}


    # all ways whose types are legal
    # trim the part of the way that is not in the space range
    for ele in root:
        if ele.tag == 'way':
            for ele2 in ele:
                if ele2.tag == 'tag': 
                    if ele2.get('k') == 'highway' and ele2.get('v') in highway_cls:
                        
                        nd_ids = []
                        for nd in ele.iter('nd'):
                            nd_ids.append(nd.get('ref'))
                        
                        if len(nd_ids) < 2:
                            break
                            
                        if nd_ids[0] in dict_nodes and nd_ids[1] in dict_nodes \
                                and in_range(dict_nodes[nd_ids[0]]['lon'], dict_nodes[nd_ids[0]]['lat']) \
                                and in_range(dict_nodes[nd_ids[1]]['lon'], dict_nodes[nd_ids[1]]['lat']):
                            
                            last_satisfied_index = 0
                            for i, nd in enumerate(nd_ids):
                                if nd in dict_nodes and in_range(dict_nodes[nd]['lon'], dict_nodes[nd]['lat']):
                                    last_satisfied_index += 1
                                else:
                                    break
                            nd_ids = nd_ids[0: last_satisfied_index]
                                                        
                        elif nd_ids[-1] in dict_nodes and nd_ids[-2] in dict_nodes \
                                and in_range(dict_nodes[nd_ids[-1]]['lon'], dict_nodes[nd_ids[-1]]['lat']) \
                                and in_range(dict_nodes[nd_ids[-2]]['lon'], dict_nodes[nd_ids[-2]]['lat']):
                            
                            first_satisfied_index = 0
                            for i in range(len(nd_ids) - 1, -1, -1):
                                if nd_ids[i] in dict_nodes and in_range(dict_nodes[nd_ids[i]]['lon'], dict_nodes[nd_ids[i]]['lat']):
                                    first_satisfied_index -= 1
                                else:
                                    break
                            nd_ids = nd_ids[first_satisfied_index: ]
                        else:
                            break

                        # get all tags
                        tags = {}
                        for _ele2 in ele:
                            if _ele2.tag == 'tag':
                                tags[_ele2.get('k')] = _ele2.get('v')

                        dict_ways[ele.get('id')] = {'nodes': nd_ids, **tags}

                        #
                        for nd in nd_ids:
                            _dict_nodes_legal[nd] = dict_nodes[nd]
    
    dict_nodes = _dict_nodes_legal # remove nodes that do not appear in segments


def OSM_stats():
    global dict_ways

    way_type_counts = copy.deepcopy(highway_cls)
    for k in way_type_counts:
        way_type_counts[k] = 0
    segment_type_counts = copy.deepcopy(way_type_counts)


    for way in dict_ways.values():
        if 'highway' in way:
            _is_oneway = True if 'oneway' in way and is_oneway(way['oneway']) else False

            way_type_counts[way['highway']] += (1 if _is_oneway else 2)
            segment_type_counts[way['highway']] += ( (len(way['nodes']) - 1) * (1 if _is_oneway else 2) )

    print('===directed ways===' + str(sum(list(way_type_counts.values()))) )
    print(way_type_counts)
    print('===directed segments==='+ str(sum(list(segment_type_counts.values()))))
    print(segment_type_counts)


def output_node_files(node_file):
    global dict_nodes
    # node
    with open(node_file, 'w', encoding = 'UTF-8') as f_handle:
        f_handle.write('node_id,lon,lat\n')
        
        for node_id, values in dict_nodes.items():
            line = node_id + ',' + \
                    str(values['lon']) + ',' + \
                    str(values['lat']) + '\n'
            f_handle.write(line)


def output_segment_way_files(segment_file, way_file, adj_segment_file):
    global dict_nodes
    global dict_ways
    global dict_segments
    global dict_segments_way_ids

    # segment & way & adj_segments
    with open(segment_file, 'w', encoding = 'UTF-8') as f_handle_seg, \
            open(way_file, 'w', encoding = 'UTF-8') as f_handle_way, \
            open(adj_segment_file, 'w', encoding = 'UTF-8') as f_handle_adjseg:
        
        dict_node_inseg = {} # ingress nodes
        dict_node_outseg = {} # outgress nodes 

        f_handle_seg.write('s_id,e_id,s_lon,s_lat,e_lon,e_lat,c_lon,c_lat,length,radian,highway_cls,way_ids,bridge,maxspeed,lanes,inc_id\n') # directed
        f_handle_way.write('way_id,node_ids,highway,highway_cls,one_way,bridge,maxspeed,lanes\n') # undirected but has one-way flag
        f_handle_adjseg.write('s1s_id,s1e_id,s2s_id,s2e_id,s_id,e_id,weight,s1_length\n')

        for way_id, values in dict_ways.items():
            # way_id = int(way.get('id'))
            node_id_seq = values['nodes'] # id type: string; [id1, id2, id3...]
            node_seq_dup = [] # id type: object; [id1, id2, id2, id3, id3, id4...]

            for nd1, nd2 in pairwise(node_id_seq):
                node_seq_dup.extend([nd1, nd2])

            # is single way
            _is_oneway = True if 'oneway' in values and is_oneway(values['oneway']) else False
            if 'oneway' in values and str(values['oneway']) == '-1': # rarely happens
                node_id_seq.reverse()
                node_seq_dup.reverse()

            _bridge = values['bridge'] if 'bridge' in values else 'NG'
            _maxspeed = values['maxspeed'] if 'maxspeed' in values else 'NG'
            _lanes = values['lanes'] if 'lanes' in values else 'NG'

            # output way data
            line = str(way_id) + ',' + \
                    '"[' + \
                    ','.join(str(it) for it in node_id_seq) + \
                    ']"' + ',' + \
                    values['highway'] + ',' + \
                    str(highway_cls[values['highway']]) + ',' + \
                    ('1' if _is_oneway else '0') + ',' + \
                    _bridge + ',' + \
                    _maxspeed + ',' + \
                    _lanes + '\n'
            f_handle_way.write(line)


            def _edge_to_segs(seq, symbol): 
                for _i in range(0, len(seq), 2):
                    last_node = seq[_i] # string id
                    cur_node = seq[_i+1] # string id

                    if last_node in dict_node_outseg:
                        dict_node_outseg[last_node].add(cur_node)
                    else:
                        dict_node_outseg[last_node] = {cur_node}

                    if cur_node in dict_node_inseg:
                        dict_node_inseg[cur_node].add(last_node)
                    else:
                        dict_node_inseg[cur_node] = {last_node}


                    if (last_node,cur_node) in dict_segments:
                        dict_segments_way_ids[(last_node,cur_node)].append(way_id)
                    else:
                        dict_segments[(last_node,cur_node)] = {}
                        dict_segments_way_ids[(last_node,cur_node)] = [way_id]


            _edge_to_segs(node_seq_dup, 0)

            if not _is_oneway: # bi-way
                node_seq_dup.reverse()
                _edge_to_segs(node_seq_dup, 5000)
    
        # may have duplicates, hence output here.
        _inc_id = 100

        # output segments
        for (last_node, cur_node) in dict_segments:
            _length = haversine(dict_nodes[last_node]['lon'], dict_nodes[last_node]['lat'],
                    dict_nodes[cur_node]['lon'], dict_nodes[cur_node]['lat'])
            _radian = radian(dict_nodes[last_node]['lon'], dict_nodes[last_node]['lat'],
                    dict_nodes[cur_node]['lon'], dict_nodes[cur_node]['lat'])
            _way_value = dict_ways[dict_segments_way_ids[(last_node, cur_node)][0]]
            _highway_cls = highway_cls[_way_value['highway']]
            _bridge = _way_value['bridge'] if 'bridge' in _way_value else 'NG'
            _maxspeed = _way_value['maxspeed'] if 'maxspeed' in _way_value else 'NG'
            _lanes = _way_value['lanes'] if 'lanes' in _way_value else 'NG'

            _middle_point = ( \
                            round((dict_nodes[last_node]['lon'] + dict_nodes[cur_node]['lon']) / 2, 8), \
                            round((dict_nodes[last_node]['lat'] + dict_nodes[cur_node]['lat']) / 2, 8))
            _line = last_node + ',' + \
                    cur_node + ',' + \
                    str(dict_nodes[last_node]['lon']) + ',' + \
                    str(dict_nodes[last_node]['lat']) + ',' + \
                    str(dict_nodes[cur_node]['lon']) + ',' + \
                    str(dict_nodes[cur_node]['lat']) + ',' + \
                    str(_middle_point[0]) + ',' + \
                    str(_middle_point[1]) + ',' + \
                    str(_length) + ',' + \
                    str(_radian) + ',' + \
                    str(_highway_cls) + ',' + \
                    '"[' + \
                    ','.join(str(it) for it in dict_segments_way_ids[(last_node, cur_node)]) + \
                    ']"' + ',' + \
                    _bridge + ',' + \
                    _maxspeed + ',' + \
                    _lanes + ',' + \
                    str(_inc_id) + '\n'
            f_handle_seg.write(_line)
            dict_segments[(last_node, cur_node)]['inc_id'] = _inc_id
            _inc_id += 1

        # output adjacent segment relations
        for node_id, values in dict_nodes.items():
            mid = node_id
            left_set = dict_node_inseg.get(mid)
            right_set = dict_node_outseg.get(mid)
            
            if left_set is None or right_set is None:
                continue

            for left in left_set:
                for right in right_set:
                    if left == right:
                        if len(left_set) == 2 and len(right_set) == 2: # TODO: correct?
                            continue

                    _weight = 1.0
                    # _weight used (mid,right)
                    _way_values1 = dict_ways[dict_segments_way_ids[(left,mid)][0]]
                    _way_values2 = dict_ways[dict_segments_way_ids[(mid,right)][0]]
                    if 'highway' in _way_values1 and _way_values1['highway'] in highway_cls \
                            and 'highway' in _way_values2 and _way_values2['highway'] in highway_cls:
                        _weight = get_highway_weight(_way_values1['highway']) + get_highway_weight(_way_values2['highway'])
                    if left == right: # u-turn
                        _weight /= 10 # TODO

                    _length = haversine(dict_nodes[left]['lon'], dict_nodes[left]['lat'],
                            dict_nodes[mid]['lon'], dict_nodes[mid]['lat'])

                    line = str(left)  + ',' + \
                            str(mid)  + ',' + \
                            str(mid)  + ',' + \
                            str(right)  + ',' + \
                            str(dict_segments[(left,mid)]['inc_id']) + ',' + \
                            str(dict_segments[(mid,right)]['inc_id']) + ',' + \
                            str(_weight) + ',' + \
                            str(_length) + '\n'
                    f_handle_adjseg.write(line)


def output_cell_files_new(cell_file):
    global dict_nodes
    global dict_ways
    global dict_segments
    global dict_segments_way_ids
    global dict_cells_by_index

    # ===cell===
    # range of the space
    lon_min, lon_max, lat_min, lat_max = 99999, -99999, 99999, -99999
    for node_id, values in dict_nodes.items():
        _lon = values['lon']
        _lat = values['lat']
        lon_min = _lon if _lon < lon_min else lon_min
        lon_max = _lon if _lon > lon_max else lon_max
        lat_min = _lat if _lat < lat_min else lat_min
        lat_max = _lat if _lat > lat_max else lat_max

    print('space range: {}'.format([lon_min, lon_max, lat_min, lat_max]))

    lon_size = math.ceil((lon_max - lon_min) / lon_unit)
    lat_size = math.ceil((lat_max - lat_min) / lat_unit)
    print('number of cells: {}*{} [lon*lat]'.format(lon_size, lat_size))

    # cellspace obj & output
    cellspace = CellSpace(lon_unit, lat_unit, lon_min, lat_min, lon_max, lat_max) 
    with open(cell_file + 'space_info.pickle', 'wb') as f_cellspace_info:
        pickle.dump(cellspace, f_cellspace_info, protocol = pickle.HIGHEST_PROTOCOL)


# output data files
def output_dataset_files(output_file_name_prefix):
    node_file = output_file_name_prefix + '_node'

    segment_file = output_file_name_prefix + '_segment'
    way_file = output_file_name_prefix + '_way'
    adj_segment_file = output_file_name_prefix + '_adjsegment'

    cell_file = output_file_name_prefix + '_cell'

    output_node_files(node_file)

    output_segment_way_files(segment_file, way_file, adj_segment_file)

    output_cell_files_new(cell_file)



'''
Given a downloaded OSM data file, generating corresponding data files that are required in project.

Genereated data files (csv):
1. xxx_rn.          removing illegal nodes and roads, based on the original raw data.
2. xxx_node.        all nodes.
3. xxx_segment.     all segments. (p.s. a way is composed of multiple segments.)
4. xxx_way.         all ways.
5. xxx_adjsegment.  adjacent segments. (seg1 -> seg2)
6. xxx_cell.        
7. xxx_adjcell.     


[Example]
python osm2roadnetwork.py ../data/OSM_Chengdu_2thRings_raw

[Argv]
    argv[1] = raw download osm file
'''
if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(-1)

    # load area range
    ___ring_ranges = [kv[1] for kv in rings_ranges.items() if sys.argv[1].split('/')[-1] == kv[0]]
    assert len(___ring_ranges) == 1
    ring_ranges = ___ring_ranges[0]

    ___lonlat_unit = [kv[1] for kv in lonlat_units.items() if sys.argv[1].split('/')[-1] == kv[0]]
    lonlat_unit = ___lonlat_unit[0]
    lon_unit = lonlat_unit['lon_unit']
    lat_unit = lonlat_unit['lat_unit']

    read_OSM_file(sys.argv[1])
    print(len(dict_nodes), len(dict_ways), len(dict_legal_nodes))
    
    OSM_stats()

    output_dataset_files(sys.argv[1])

    