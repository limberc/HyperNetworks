<div align="center">    

# HyperNetworks

[![Paper](http://img.shields.io/badge/paper-arxiv.1609.09106-B31B1B.svg)](https://arxiv.org/abs/1609.09106)
ARXIV
</div>

## Description

Static HyperNetworks Implementation with Principal Weighted Initialization on ResNet.

## How to run

First, install dependencies

```bash
# clone project   
git clone https://github.com/limberc/HyperNetworks

# install project   
cd HyperNetworks
pip install -r requirements.txt
 ```   

Next, navigate to any file and run it.

 ```bash
# module folder
cd hypernet

# run module
python train.py --dataset {cifar10/cifar100} --gpus $num_gpu
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from hypernet.resnet import resnet18
from pytorch_lightning import Trainer
from torchvision.datasets import CIFAR10, CIFAR100

# model
model = resnet18()

# data
train, val, test = CIFAR100()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation

```
@article{ha2016hypernetworks,
  title={Hypernetworks},
  author={Ha, David and Dai, Andrew and Le, Quoc V},
  journal={arXiv preprint arXiv:1609.09106},
  year={2016}
}
```   
