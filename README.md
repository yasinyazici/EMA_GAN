# EMA_GAN

This is the original implementation of [The Unusual Effectiveness of Averaging in GAN Training](https://arxiv.org/abs/1806.04498). The code is adapted from https://github.com/pfnet-research/chainer-gan-lib.

A sample code:

```python train.py --dataset='cifar10' --objective='gan' --n_dis=1 --model='base' --gpu=0 &```
