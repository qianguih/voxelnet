# Introduction

This is an unofficial inplementation of [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396) in TensorFlow. A large part of this project is based on the work [here](https://github.com/jeasinema/VoxelNet-tensorflow). Thanks to [@jeasinema](https://github.com/jeasinema). This work is a modified version with bugs fixed and better experimental settings to chase the results reported in the paper (still ongoing).

# Dependencies
- `python3.5+`
- `TensorFlow` (tested on 1.4.1)
- `opencv`
- `shapely`
- `numba`

# Installation
1. Clone this repository.
2. Compile the Cython module
```bash
$ python setup.py build_ext --inplace
```
3. Compile the evaluation code
```bash
$ cd kitti_eval
$ g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
```

# Data Preparation
1. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Data to download include:
    * Velodyne point clouds (29 GB): input data to VoxelNet
    * Training labels of object data set (5 MB): input label to VoxelNet
    * Camera calibration matrices of object data set (16 MB): for visualization of predictions
    * Left color images of object data set (12 GB): for visualization of predictions

2. In this project, we use the cropped point cloud data for training and testing. Point clouds outside the image coordinates are removed. Update the directories in `data/crop.py` and run `data/crop.py` to generate cropped data. Note that cropped point cloud data will overwrite raw point cloud data here.

2. Split the training set into training and validation set according to the protocol [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz). And rearrange the training dataset to have the following structure:
```plain
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
       └── evaluation  <--- evaluation data
       ├── image_2
       ├── label_2
       └── velodyne
       ```
        
3. Update the dataset directory in `config.py`

# Train
1. Specify the GPUs to use in `config.py`
2. run `train.py` with desired hyper-parameters to start training:
```bash
$ python3 train.py
```

# Evaluate
1. run `test.py` to produce predictions on validation set.
```bash
$ python3 test.py
```
results will be dumped into `predictions/data`. Set the `-vis` flag to True if dumping visualizations and they will be saved into `predictions/vis`. Currently, setting `-vis` will evoke a deadlock problem and requires to manully kill the process when testing is done.

2. run the following command to measure quantitative performances of predictions:
```bash
$ ./kitti_eval/evaluate_object_3d_offline [KITTI_DATA]/evaluation/label_2 ./predictions
```

# Performances

##### AP

|  | Easy | Moderate | Hard |
|:-:|:-:|:-:|:-:|
| Reported | 81.97 | 65.46 | 62.85 |
| Reproduced | 64.39  | 57.87 | 58.35 |


# TODO

- [ ] fix the deadlock problem in multi-thread processing in training
- [ ] fix the deadlock problem when `vis` flag is set to True in `test.py`
- [ ] replace averaged calibration matrices with correct ones

