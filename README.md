## SARN: Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning

This is a pytorch implementation of the [SARN paper](https://openproceedings.org/2023/conf/edbt/paper-193.pdf):

```
@inproceedings{ChangTC023,
  author    = {Yanchuan Chang and Egemen Tanin and Xin Cao and Jianzhong Qi},
  title     = {Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning},
  booktitle = {Proceedings 26th International Conference on Extending Database Technology, {EDBT}},
  pages     = {144--156},
  year      = {2023},
}
```


### Requirements
- Ubuntu 20.04 LTS with Python 3.7.7
- `pip install -r requirements.txt`
- Download SF dataset here, and unzip it into `./data`.


### Quick Start
First pre-train a SARN (cf. Section Self-supervised Training), then it can be used in any downstream task, where the parameters can be fine-tuned or frozen (cf. Section Downstream Task Prediction).

#### Self-supervised Training

Pre-train SARN and run *road property prediction* task with the frozen embeddings. The trained SARN and the corresponding learned embeddings are persisted to disk for other downstream tasks. Logs are ouputted to the terminal and dumped to the log files in `./exp/log`. 
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name classify
```

#### Downstream Task Prediction

We focus on three downstream tasks on road networks, including *road property prediction*, *trajectory similarity prediction* and *shortest-path distance prediction* - **classify**, **trajsimi** and **spd** for short, respectively. 

Fine-tune the pre-trained SARN model and train classify task model. (Prerequisites: a pre-trained SARN model, in other words, run the last command first).
```bash
python train.py --task_encoder_model SARN_ft --dataset SF --task_name classify --task_pretrained_model
```

Fine-tune the pre-trained SARN model and train trajsimi task model.
```bash
python train.py --task_encoder_model SARN_ft --dataset SF --task_name trajsimi --task_pretrained_model 
```

Fine-tune the pre-trained SARN model and train spd task model.
```bash
python train.py --task_encoder_model SARN_ft --dataset SF --task_name spd --task_pretrained_model 
```

Train classify task model with the frozen embeddings.
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name classify --task_pretrained_model 
```

Train trajsimi task model with the frozen embeddings.
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name trajsimi --task_pretrained_model 
```

Train spd task model with the frozen embeddings.
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name spd --task_pretrained_model 
```


### FAQ
##### Datasets
To use your own datasets, you may need to follow the steps below:
1. Download the [OpenStreetMap](https://www.openstreetmap.org/) xml dataset of a certain area.
2. Extract the road network from the xml dataset and dump the data to the dedicated files. (See `./utils/osm2roadnetwork.py` and `./data/OSM_SanFrancisco_downtown_raw`).
3. Prepare the trajectory dataset used in trajsimi task. (See `./utils/traj_preprocess_sf.py`). 
4. The ground-truth labels used in downstream tasks are created during the downstream task training, but such process will be only executed once. After the first time, such labels will be read from files created during the first execution. (See `./task/classifier.py::Classifier.classifier_datasets`, `./task/traj_simi_v2.py::TrajSimi.load_trajsimi_dataset` and `./task/spd.py::SPD.get_spd_dict`).

##### A potential issue
(Let \$the\_list = [torch-scatter,torch-sparse,torch_geometric,torch-cluster,torch-spline-conv].)  
Commonly, it will take a long time (~10 minutes) to install the packages in \$the_list since their wheels are built locally. If you fail to directly install the packages in \$the_list via the pip command provided ahead, please follow the instruction [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels) to install the five packages in \$the\_list via pip wheels.



### Contact
Email changyanchuan@gmail.com if you have any inquiry.
