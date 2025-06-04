# SRNN: Reconstructing Surface by Predicting Nearest Neighbors

> This is the official implementation of *SRNN: Reconstructing Surface by Predicting Nearest Neighbors*.

This project implements the data preparation, surface reconstruction, and metrics evaluation described in our paper.

## Environment

We provide a Dockerfile to create a docker image containing all training/testing/visualization tools used in this project. Run
```shell
docker buildx build -t srnn -f Dockerfile .
```
to build the `srnn` image. This image has 3 mount points for mounting data/code directories:
- `/root/SRNN` for this codebase
- `/root/data` for pointcloud & mesh data
- `/root/prior-data` for prior database

An example of starting a container:
```shell
docker run -itd --name reconstruction --shm-size 8g --gpus all -v /data:/root/data -v /ssd/data:/root/prior-data -v /projects/SRNN:/root/SRNN srnn /bin/bash
```
Then run `docker exec -it reconstruction bash` to attach to the container.

If you prefer to develop or run code without docker, please install Python 3.10 and `cuda 12.4` with Anaconda, then run `pip install -r requirements.txt` to create a virtual environment locally.

## Dataset

### Preparing Files

We use the ShapeNetCore.v1 dataset for all training/testing tasks. Because this dataset is public available and we have not add additional meshes, we only release the data processing scripts so that you can process the dataset by yourself easily.

First, please unzip the category packs you want to use (e.g. *02691156.zip* for the airplane category), putting all categories into one directory like this:
```
ShapeNetCore
├──02691156 # A category
│  ├──...
│  └──de5807cb73bcbbf18587e940b916a22f # An object
│     ├──model.obj
│     ├──model.mtl
│     └──(maybe some texture images)
├──02828884
├──...
└──04530566
```

Then run the script *dataset/remove_shapenet_textures.py* to delete all texture images in the dataset, because some PNG images are named with a `.jpg` suffix, causing Open3D crashing when reading the `.mtl` files. This script only uses Python built-in packages, you are not need to run it with docker.

### Pre-Processing

Now you can run *dataset/sample_pointcloud.py* to sample
- Trining/test dataset for the Nearest Neighbor Prior (query points, KNNs and nearest neighbors)
- Point cloud from each mesh

```shell
PYTHONPATH=. python3 dataset/sample_pointcloud.py --n-pcd-points 2048 \
    --n-query-points 2048 \
    -K 100 \
    --prior --reconstruction \
    [path to your ShapeNetCore dataset] \
    [path to save the pointcloud files and the prior database]
```

> For more information about the command-line arguments, please run `PYTHONPATH=. python3 dataset/sample_pointcloud.py --help`.
>
> It is strongly recommended to **use an SSD instead of HDD** to save the prior dataset, because the script will shuffle a loooot of lines in the database (performing a lot of random I/O), such an operation may take a week on HDD. You can move all sampling results to an HDD after all sampling tasks are done, but it is still recommended to keep the prior database on your SSD at least.

Once sampling is done, you can find the pointcloud (as `.ply` files) and a prior database (as `.db` file) in the path you specified like this:
```
[path to save the pointcloud files and the prior database]
├──02691156 # A category
│  ├──...
│  ├──de5807cb73bcbbf18587e940b916a22f.ply # Some PLY files
│  ├──train.txt # The training list
│  └──test.txt  # The test list
├──02828884
├──...
├──04530566
├──prior-2048-100-tmp.db
└──prior-2048-100.db
```
Within each category, there will be a *train.txt* (and a *test.txt*) containing all full\_ID of objects for training (or testing) the prior. The *prior-2048-100-tmp.db* is a temporary file used for shuffling, you can delete this file once you have already got the *prior-2048-100.db*.

> By default, points in each KNN are sorted according to the distance between itself and the query point. So if you have sampled *prior-2048-100.db* (i.e. K=100), you can also easily obtain KNNs with 50 points (i.e. K=50) without re-sampling. The *dataset/prior.py* allows you specify `K` when loading KNN.

To generate a subset of pointclouds for reconstruction test, run
```shell
PYTHONPATH=. python3 dataset/generate_subset.py --name subtest -N 100 [path to the pointcloud dataset]
```
This script will only output a `.txt` file like *train.txt*, you can specify the filename through the `--name` argument and the number of objects by the `-N` argument. By default, the script will generate a *subtest.txt* for each category, containing full ID of 100 objects.

## Train the Nearest Neighbor Prior

To train the prior function (network), run
```shell
python3 train_prior.py --database [path to the prior-2048-100.db]
```
You can also modify the default learning rate/epochs/K by passing command-line arguments. For further information, please run `python3 train_prior.py --help`. Weights of the prior network will be saved in *pretrained-models* after each epoch. The script will save all logging messages to *log/prior*, including a text file and the tensorboard file.

## Surface Reconstruction

The main entry point of surface reconstruction is *reconstruct.py*, this script can be used for
- exploring single object reconstruction, with detailed tensorboard logging information
- used by another script for batched tasks

For example, you can reconstruct surface from *examples/airplane.ply* supervised by the prior model *k100-pn-w512-d3.pth*:
```shell
python3 reconstruct.py -P pretrained-models/k100-pn-w512-d3.pth --tensorboard --progress-bar --save-progress examples/airplane.ply results/airplane.obj
```
It will show a progress bar on the terminal, log loss values and SDF slices to tensorboard files (in *log/reconstruction*) and save intermediate results in *results/progress*. For more explanation about all command-line arguments, please run `python3 reconstruct.py --help`.

If you want to perform reconstruction on the subset generated by *dataset/generate_subset.py*, run
```shell
python3 test_shapenet.py --gpu 0,1,2,3 --tasks-per-gpu 1 --name [name of the subset file] [path to the pointcloud dataset] [other arguments for reconstruct.py]
```
This script will execute multiple reconstruction task on GPUs specified by `--gpu` (in the above case, GPU 0, 1, 2, 3 are used). Any argument passed after `[path to the pointcloud dataset]` will be directly passed to *reconstruct.py* so you can specify hyper-parameters as if you are directly using *reconstruct.py*.

*test_shapenet.py* will save reconstructed mesh to the *results* directory, you should see a folder named by the timestamp at the beginning of reconstruction task, e.g. *results/2025-03-25T22.14.07* for a task started at 22:14:07 on 2025-03-25.

## Evaluating Metrics

Use the script *evaluate_metrics.py* to evaluate CD, F-Score and NC (Normal Consistency) of the reconstruction results.

Assume you have reconstruction results saved in *results/2025-03-35T22.14.07*, you can run
```shell
python3 evaluate_metrics.py [path to the shapenet dataset] results/2025-03-25T22.14.07
```
to evaluate all metrics (using GPU 0). Once the evaluation has done, there will be
- A *stats.csv* in the result folder for average CD/F-Score/NC on each category
- A *metrics.csv* in each category folder for CD/F-Score-NC of each object

An additional argument `--nc-vis-path` can be used to specify a path to save colored pointclouds for visualizing normal consistency. Each point will be painted with a color from pure blue (best NC) to pure red (worst NC). Note: saving these colored pointclouds will use 20+GiB disk space and consume more time.

> Note: Due to the C++ implementation, the script *evaluate_metrics.py* must be run on CUDA device 0, or an error (illegal memory access) will occur.

## Rendering & Visualization

We use Blender for rendering & 3D visualization. By default, `bpy 4.0.0` is installed in the docker container. If you want to use your own Blender (it's OK because the blender scripts does not rely on PyTorch), please ensure your Blender version is equal or newer than 4.0.

For 3D visualization, see [Instructions for Rendering](./blender-scripts/README.md).

## Further Analysis

To sample query points for analyzing tangent error distribution, use the scripts under *analysis*:
```shell
PYTHONPATH=. python3 analysis/tangent_error.py \
    --prior-model pretrained-models/k100-pn-w512-d3.pth \
    --n-query-points 10000 \
    --n-offset-points 10 \
    -K 100 \
    --gpu 0 \
    [path to the ShapeNetCore dataset] \
    [path to the ShapeNetCore point cloud dataset]
```

The tangent error values will be saved in `.npz` files in *results/tangent-error*, each file contains two NumPy arrays:
- `dist`: Distance from the query point to the point cloud.
- `error`: Tangent error of the query point.

Then you can perform any statistical analysis on the error values.

If you want to visualize the sampled query points (with their tangent disks), run `PYTHONPATH=. python3 analysis/show_tangent_samples.py [mesh file] [point cloud file]` to produce a *query.ply* file and render it with the GT mesh together using Blender.
