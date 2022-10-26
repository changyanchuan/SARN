import os
import sys
import logging
import pickle
import argparse

from config import Config as Config
from utils import tool_funcs
from utils.osm_loader import OSMLoader

from model.SARN import SARN
from task.classifier import Classifier
from task.traj_simi_v2 import TrajSimi
from task.spd import SPD


def get_encoder(name):
    if name == 'SARN' or name == 'SARN_ft':
        return SARN
    else:
        exit(-10001)


def get_task(name):
    if name == 'classify':
        return Classifier
    elif name == 'trajsimi':
        return TrajSimi
    elif name == 'spd':
        return SPD
    else:
        exit(-10002)


def get_encoder_dump_filehandle(name):
    f_path = ''
    if name == 'SARN':
        f_path = "{}/exp/snapshots/{}_SARN_best_embs{}.pickle".format(
                Config.root_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)
    else:
        exit(-10003)
    
    if not os.path.exists(f_path):
        exit(-10004)

    return open(f_path, 'rb')


def get_encoder_checkpoint_filepath(name):
    f_path = ''
    if name == 'SARN_ft':
        f_path = "{}/exp/snapshots/{}_SARN_best{}.pkl".format(
                Config.root_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)
    else:
        exit(-10003)

    if not os.path.exists(f_path):
        exit(-10004)
    return f_path


def parse_args():
    parser = argparse.ArgumentParser(description = "...")
    # dont give default value here! Otherwise, they will be overwritten by the values in config.py.
    # config.py is used for setting the default values

    parser.add_argument('--dataset', type = str, required = True, help = 'SF|CD|BJ')
    parser.add_argument('--task_pretrained_model', dest = 'task_pretrained_model', action='store_true')
    parser.add_argument('--task_name', type = str, required = True, help = 'classify|trajsimi|spd')
    parser.add_argument('--task_encoder_model', type = str, required = True, help = 'dump|finetune')

    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():

    encoder_name = Config.task_encoder_model
    Encoder = get_encoder(encoder_name)

    task_name = Config.task_name
    Task = get_task(task_name)

    metrics = tool_funcs.Metrics()

    embs = None   

    if Config.task_pretrained_model:
        if Config.task_encoder_mode == 'dump':
            # load seg embeddings from pre-trained models
            fh = get_encoder_dump_filehandle(encoder_name)
            embs = pickle.load(fh).to(Config.device)
            fh.close()
            
            osm_data = OSMLoader(Config.dataset_path)
            osm_data.load_data()
            tsk = Task(osm_data, embs, None)
            metrics.add(tsk.train())

        elif Config.task_encoder_mode == 'finetune':
            enc = Encoder()
            f_path = get_encoder_checkpoint_filepath(encoder_name)
            enc.load_model_state(f_path)
            osm_data = OSMLoader(Config.dataset_path)
            osm_data.load_data()

            tsk = Task(osm_data, None, enc)
            metrics.add(tsk.train())
    else: # not pre-trained
        # first, train and obtain embeddings, then, do downstream task
        enc = Encoder()
        metrics.add(enc.train())

        if Config.task_encoder_mode == 'dump':
            embs = enc.get_embeddings(True)
            enc.dump_embeddings(embs) 
            tsk = Task(enc.osm_data, embs, None)

        elif Config.task_encoder_mode == 'finetune':
            f_path = get_encoder_checkpoint_filepath(encoder_name)
            enc.load_model_state(f_path)
            tsk = Task(enc.osm_data, None, enc)

        metrics.add(tsk.train())

    logging.info('[EXPFlag]task={},model={},dataset={},{}'.format(
                Config.task_name, Config.task_encoder_model, Config.dataset, str(metrics)))


if __name__ == '__main__':
    Config.update(parse_args())
    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('pid: ' + str(os.getpid()))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')
  
    main()