# Jet SSD

This repository contains components of the Jet SSD study. Feel free to reproduce the results and report issues if any. Check slides [here](https://indico.cern.ch/event/948465/contributions/4324196/attachments/2248038/3813115/JetSSD-vCHEP2021.pdf) and read more [here](https://arxiv.org/abs/2105.05785).

## Setup
The environment is specified in the `environment.yml` file. It can be setup like this:
```
conda env create -f misc/environment.yml
conda activate jetssd
```

The environment is quite big. Depending on which part of the study needs reproduction, some of the packages may be unnecessary.

Also download color palette with `sh scripts/get-color-palette.sh`.

## Dataset

### Option 1: Download
For the fastest and simplest setup, download files from [Zenodo](https://doi.org/10.5281/zenodo.4883651).

### Option 2: Generate or access

#### (optional) `ROOT` files
Depending on your setup, it may be possible to access the files from CERN `eos`. The files are located in `/eos/project/d/dshep/CEVA/`. Run your ROOT simulation in case these files are inaccessible with the card from [here](https://github.com/AdrianAlan/jet-ssd/blob/master/misc/CMS_PhaseII_50PU_CEVA.tcl).

#### (optional) Full `HDF5` files
Files are available in `/eos/project/d/dshep/CEVA/`. If not accessible, to generate `HDF5` files for all the available `ROOT` data, replace `<src>` with `/eos/project/d/dshep/CEVA/` or the destination directory from the previous step. First, run
```
mkdir data
python scripts/generate-configuration-file.py <src> data/file-configuration.json
```
to generate information about file content (files may be irregular when there were failures during simulation). The configuration file will be stored in the `data` folder.

To generate `HDF5` files replace `<src>` with source dirtectory , e.g. `/eos/project/d/dshep/CEVA/RSGraviton_WW_NARROW` and `<dst>` with output directory. This line will output one file for each `<src>` directory.
```
python hdf5-generator.py <src> -o <dst> -v
```

#### Target size `HDF5` files
Depending on your setup, it may be possible to access the files from CERN `eos`. The files are located in `/eos/project/d/dshep/CEVA-hdf5/mix`. If not accessible, please run the optional steps from above first.

To run, replace `dataset_misc` section in `ssd-config.yml` with proper source directory name (`<dst>` from previos step or `/eos/project/d/dshep/CEVA-hdf5`) and target sizes. Lastly, change `dataset` section in `ssd-config.yml` with proper target directory paths for `HDF5` files, e.g. replace `/path/to/train0` with `./foo/my-train-0.h5`. The following line will generate target files.
```
python hdf5-generator-mix.py
```

## (optional) Explore the data
Two notebooks help with understanding and verifying the data.
* `a-walk-through` gives a general introduction.
* `profile-dataset` shows dataset details.

## Training and evaluation
Prerequisite:
* Change `ssd-config.yml` file paths in the `dataset` to point to your `HDF5` files.

Run training with:
* `python jet-ssd-train.py <name_fpn> -v` for full precision,
* `python jet-ssd-train.py <name_int8> -m <path_fpn> -v8` for INT8 precision,
* `python jet-ssd-train.py <name_twn> -m <path_fpn> -vt` for ternary weights.

Run evaluation with (all plots will be stored in `plots` directory):
```
python jet-ssd-eval.py <name_fpn> <name_twn> <name_int8> -v
```

Other notebooks:
* Verify inference by running the `ssd-inference` notebook,
* Check ternary filters by running the `show-filters` notebook.

## Pruning
Prerequisite:
* Change `ssd-config.yml` `learning_rate` and `max_epochs`.

To run training with regularizer, add `-r` flag:
* `python jet-ssd-train.py <name_fpn> -rv -s <path_to_net-config> -m <path_to_previous_iteration>`

Uncomment target `max_channels` in `jet-ssd-prune.py` and run:
* `python jet-ssd-prune.py <name_fpn> -rv -s <path_to_net-config>`
This will produce a new `net-config.yml` file and overwrite the input model. You may want to make copies of your models before running the pruning procedure.

## Inference tests

* Convert model to ONNX with `python jet-ssd-onnx-export.py <name> -v`.
* Measure Jet SSD inference time on CPUs GPUs with `python jet-ssd-benchmark-inference.py <name> -b <batch_size> -v`. Additinal flag `--onnx` for ONNX runtime and `--trt` for TensorRT. Without a flag, the tests will run for PyTorch.
* To get the final inference plot, change the data in `jet-ssd-inference.py` and run.

## Plots
All plots will be saved in the `plots` folder unless  otherwise specified in `ssd-config.yml`.

## Pretrained Checkpoints
* [FPN](https://drive.google.com/file/d/1NjRNcnGsvHzibn_rVdbgj2hL1Sk780oK/view?usp=sharing)
* [TWN](https://drive.google.com/file/d/1LGF00JaS6JYU4GEBYv88-zg2e4Z2stPp/view?usp=sharing)
* [INT8](https://drive.google.com/file/d/1K1RjppW_yNAbLXYMxHw4DnM0s3sjMNQp/view?usp=sharing)
