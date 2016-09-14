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
If you want to use NVIDIA GPUs to accelerate trainig and testing, you will also need to install following packages:
```
$ luarocks install cutorch
$ luarocks install cunn
```

To make things even faster, you can also install NVIDIA cuDNN libraries. You should: 
* Install cuDNN (version R5 EA)
* Have at least CUDA 7.0
* Have `libcudnn.so` in your library path (Install it from https://developer.nvidia.com/cuDNN )

Then install [`cudnn`](https://github.com/soumith/cudnn.torch) package:
```
$ luarocks install cudnn
```

# Train your own model


# Test the model


# License
