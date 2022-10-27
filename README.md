# SARN: Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning


![model_overview](https://user-images.githubusercontent.com/9978126/198295822-e880ff26-9d76-4531-a6ee-25dce5ec1cfd.png)


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


## Requirements
- Ubuntu 20.04 LTS with Python 3.7.7
- `pip install -r requirements.txt`
- Download SF dataset [here](https://drive.google.com/drive/folders/1mQXjGl8zi1TtBm2IKzP414GX6ID1b6Hm?usp=sharing), and `unzip -j SARN_dataset.zip -d data`


## Quick Start
First pre-train a SARN (cf. Section Self-supervised Training), then it can be used in downstream tasks, where the parameters can be fine-tuned or frozen (cf. Section Downstream Task Prediction).

### Self-supervised Training

Pre-train a SARN model and evaluate its performance on *road property prediction* task with the frozen embeddings. The trained SARN and the corresponding learned embeddings are persisted to disk (in `./exp/snapshots/`) for other downstream tasks. Logs are ouputted to the terminal and dumped to the log files in `./exp/log`. 
```bash
python train.py --task_encoder_model SARN --dataset SF
```

### Downstream Task Prediction

We focus on three downstream tasks on road networks, including *road property prediction*, *trajectory similarity prediction* and *shortest-path distance prediction* - **classify**, **trajsimi** and **spd** for short, respectively. 

Fine-tune the pre-trained SARN model and train a classify task model. (Prerequisites: a pre-trained SARN model, in other words, run the last command first).
```bash
python train.py --task_encoder_model SARN_ft --dataset SF --task_name classify --task_pretrained_model
```

Fine-tune the pre-trained SARN model and train a trajsimi task model.
```bash
python train.py --task_encoder_model SARN_ft --dataset SF --task_name trajsimi --task_pretrained_model 
```

Fine-tune the pre-trained SARN model and train a spd task model.
```bash
python train.py --task_encoder_model SARN_ft --dataset SF --task_name spd --task_pretrained_model 
```

Train a classify task model with the frozen embeddings.
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name classify --task_pretrained_model 
```

Train a trajsimi task model with the frozen embeddings.
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name trajsimi --task_pretrained_model 
```

Train a spd task model with the frozen embeddings.
```bash
python train.py --task_encoder_model SARN --dataset SF --task_name spd --task_pretrained_model 
```


## FAQ
#### Datasets
To use your own datasets, you may need to follow the steps below:
1. Download the [OpenStreetMap](https://www.openstreetmap.org/) xml dataset of a certain area. (See `./data/OSM_SanFrancisco_downtown_raw`).
2. Extract the road network from the xml dataset and dump the data to the dedicated files. (See `./utils/osm2roadnetwork.py` and `./data/OSM_SanFrancisco_downtown_raw`).
3. Prepare the trajectory dataset used in trajsimi task. (See `./utils/traj_preprocess_sf.py`). 
4. The ground-truth labels used in downstream tasks are created during the downstream task training, but such process will be only executed once. After the first time, such labels will be read from files created during the first execution. (See `./task/classifier.py::Classifier.classifier_datasets`, `./task/traj_simi_v2.py::TrajSimi.load_trajsimi_dataset` and `./task/spd.py::SPD.get_spd_dict`).



## Contact
Email changyanchuan@gmail.com if you have any inquiry.
