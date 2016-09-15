# ReviewNet
Reviewer Module for Caption Generation



# Overview

# Dependencies
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

# Data Pre-processing
For this implementation, we do not back-propagate the gradients to the CNN encoder, and extract CNN features from raw images using VGGNet.

First you can download the [MSCOCO dataset](http://mscoco.org/dataset/#download). The follow instructions will assume that you put the training/dev/test images (*.jpg files) in `data/train2014_jpg`, `data/val2014_jpg`, and `data/test2014_jpg` respectively.

Then download the VGGNet pretrained models. We would like to extract both the fc7 and conv5 features, so we have two pretrained models (the conv5 model is a subset of the fc7 model).
```
mkdir models
cd models
wget http://kimi.ml.cmu.edu/vgg_vd19_conv5.t7
wget http://kimi.ml.cmu.edu/vgg_vd19_fc7.t7
```

When the jpg files and the pretrained models are ready, we can now extract the features.
```
th feature_extractor.lua -imagePath data/train2014_jpg/ -outPath data/train2014_features_vgg_vd19_conv5/ -model models/vgg_vd19_conv5.t7
th feature_extractor.lua -imagePath data/val2014_jpg/ -outPath data/val2014_features_vgg_vd19_conv5/ -model models/vgg_vd19_conv5.t7
th feature_extractor.lua -imagePath data/test2014_jpg/ -outPath data/test2014_features_vgg_vd19_conv5/ -model models/vgg_vd19_conv5.t7
th feature_extractor.lua -imagePath data/train2014_jpg/ -outPath data/train2014_features_vgg_vd19_fc7/ -model models/vgg_vd19_fc7.t7
th feature_extractor.lua -imagePath data/val2014_jpg/ -outPath data/val2014_features_vgg_vd19_fc7/ -model models/vgg_vd19_fc7.t7
th feature_extractor.lua -imagePath data/test2014_jpg/ -outPath data/test2014_features_vgg_vd19_fc7/ -model models/vgg_vd19_fc7.t7
```


# Training

# Test


# License
