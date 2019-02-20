# EMA_GAN

This is the original implementation of [The Unusual Effectiveness of Averaging in GAN Training](https://arxiv.org/abs/1806.04498). It is also accepted to ICLR2019 (https://openreview.net/forum?id=SJgw_sRqFQ). The code is adapted from https://github.com/pfnet-research/chainer-gan-lib.

```python train.py --dataset='cifar10' --objective='gan' --n_dis=1 --model='base' --gpu=0 &```

![](/figures/ema_fig6.png)
![](/figures/ema_fig7.png)

