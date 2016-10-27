# Online Evaluation on MSCOCO Server

## Dependencies
#### Torch
The code is written in [Torch](http://torch.ch/). If you use Unix-like system, you can install it in this way:
```
# In a terminal without sudo
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh     
$ source ~/.bashrc
```

#### Torch Packages
Then we need to get some packages for Torch using [LuaRocks](https://luarocks.org/):
```
$ luarocks install nn
$ luarocks install nngraph 
$ luarocks install image 
$ luarocks install paths
```

#### GPU Support
Since we use NVIDIA GPUs to accelerate trainig and testing, you will also need to install following packages:
```
$ luarocks install cutorch
$ luarocks install cunn
```

To make things even faster, you need to also install NVIDIA cuDNN libraries. You should: 
* Install cuDNN (version R5 EA)
* Have at least CUDA 7.0
* Have `libcudnn.so` in your library path (Install it from https://developer.nvidia.com/cuDNN )

Then install [`cudnn`](https://github.com/soumith/cudnn.torch) package:
```
$ luarocks install cudnn
```

## Data Pre-processing
For this implementation, we do not back-propagate the gradients to the CNN encoder, and extract CNN features from raw images using inception-v3.

First you can download the [MSCOCO dataset](http://mscoco.org/dataset/#download). The following instructions will assume that you put the training/dev/test images (*.jpg files) in `data/train2014_jpg`, `data/val2014_jpg`, and `data/test2014_jpg` respectively.

Then download the inception-v3 pretrained models.
```
mkdir models
cd models
wget http://kimi.ml.cmu.edu/inceptionv3.net
```

When the jpg files and the pretrained models are ready, we can now extract the features.
```
th inception_feature_extractor.lua -m models/inceptionv3.net -f conv -b cudnn -i data/train2014_jpg -o data/train2014_inceptionv3_conv
th inception_feature_extractor.lua -m models/inceptionv3.net -f conv -b cudnn -i data/val2014_jpg -o data/val2014_inceptionv3_conv
th inception_feature_extractor.lua -m models/inceptionv3.net -f conv -b cudnn -i data/test2014_jpg -o data/test2014_inceptionv3_conv
th inception_feature_extractor.lua -m models/inceptionv3.net -f fc -b cudnn -i data/train2014_jpg -o data/train2014_inceptionv3_fc
th inception_feature_extractor.lua -m models/inceptionv3.net -f fc -b cudnn -i data/val2014_jpg -o data/val2014_inceptionv3_fc
th inception_feature_extractor.lua -m models/inceptionv3.net -f fc -b cudnn -i data/test2014_jpg -o data/test2014_inceptionv3_fc
```

## Training

First, we train three separate models with different random seeds
```
th main.lua -save_file_name reason_att_copy_simp_seed13.model -seed 13 -server_train_mode
th main.lua -save_file_name reason_att_copy_simp_seed23.model -seed 23 -server_train_mode
th main.lua -save_file_name reason_att_copy_simp_seed33.model -seed 33 -server_train_mode
```

Then we train an ensemble model
```
th ensemble.lua -batch_size 1 -ensemble_train_mode -save_file_name reason_att_copy_simp_ensemble13_23_33.model
```

## Evaluation

After we obtain an ensemble model, we can use beam search to generate the captions for the MSCOCO test set.  To accelerate the process, you can run it on multiple machines. For example, with 4 gpus, you can run each of the following commands on each of the machine.
```
mkdir server_test
th ensemble_beam.lua -server_test_mode -batch_num 4 -cur_batch_num 1
th ensemble_beam.lua -server_test_mode -batch_num 4 -cur_batch_num 2
th ensemble_beam.lua -server_test_mode -batch_num 4 -cur_batch_num 3
th ensemble_beam.lua -server_test_mode -batch_num 4 -cur_batch_num 4
```

The files will be stored under `server_test`. You can run
```
python gen_js.py
```
to generate the final submission file `server_test/captions_test2014_reviewnet_results.json`.

## Misc

The default configuration of hyper-parameters can be used to reproduce our results on the MSCOCO evaluation server.

Our code base uses the data splits and the evaluation scipts from https://github.com/karpathy/neuraltalk2. Some of the model implementation is inspired by https://github.com/oxford-cs-ml-2015/practical6. The inception v3 net is from https://github.com/Moodstocks/inception-v3.torch.git.

