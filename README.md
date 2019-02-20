# EMA_GAN

This is the original implementation of [The Unusual Effectiveness of Averaging in GAN Training](https://arxiv.org/abs/1806.04498). It is also accepted to ICLR2019 (https://openreview.net/forum?id=SJgw_sRqFQ).

Please reach us via emails or via github issues for any enquiries!

Please cite our work if you find it useful for your research and work:

```
@inproceedings{
yaz{\i}c{\i}2018the,
title={The Unusual Effectiveness of Averaging in {GAN} Training},
author={Yasin Yaz{\i}c{\i} and Chuan-Sheng Foo and Stefan Winkler and Kim-Hui Yap and Georgios Piliouras and Vijay Chandrasekhar},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=SJgw_sRqFQ},
}
```

![](/figures/ema_fig6.png)
![](/figures/ema_fig7.png)

 The code is adapted from https://github.com/pfnet-research/chainer-gan-lib.
 
 To run cifar10 experiments:

```python train.py --dataset='cifar10' --objective='gan' --n_dis=1 --model='base' --gpu=0 &```

You might need to change folder paths etc.

