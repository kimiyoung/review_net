# Offline Evaluation on MSCOCO Image Captioning

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

We use the same data splits as http://arxiv.org/abs/1502.03044 in the following sections. We do early stopping on the dev set, and evaluate the models on the test set.

## Training

There are several models available in this repository. `soft_att_lstm` refers to the [Soft Attention model](http://arxiv.org/abs/1502.03044), `reason_att` refers to the [Encode-Reivew-Decode (ERD) model](https://arxiv.org/abs/1605.07912), and `reason_att_copy` refers to the ERD model with untied weights. In our experiments, `reason_att_copy` gives the best performance.

To train a model, run
```
th main.lua -model_pack <model> -save_file -save_file_name <filename>
```
where `<model>` can be `soft_att_lstm`, `reason_att`, or `reason_att_copy`, and `<filename>` is a path to the filename for saving the trained models. The model that performs the best on the dev set will be saved.

## Test

To test a model with beam search, run
```
th <model>_eval.lua -load_file -load_file_name <filename> -test_mode
```
where `<model>` can be `soft_att_lstm`, `reason_att`, or `reason_att_copy`, and `<filename>` is a path to the filename for loading the trained models. The option `-test_mode` is telling the data loader to take care of data splits.

To test a model with greedy search, run
```
th main.lua -model_pack <model> -load_file -load_file_name <filename> -test_mode -LR 0
```
where we set the learning rate at 0 to obtain an evaluation. You have to manually stop the program after the first iteration.

## Misc

The default configuration of hyper-parameters can be used to reproduce the results in our paper for the offline evaluation experiments. For more options, please refer to `opts.lua` for more details.

Our code base uses the data splits from https://github.com/kelvinxu/arctic-captions, and the evaluation scipts from https://github.com/karpathy/neuraltalk2. Some of the model implementation is inspired by https://github.com/oxford-cs-ml-2015/practical6.

