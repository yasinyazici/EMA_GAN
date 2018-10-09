import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from common.sn.sn_linear import SNLinear
from common.sn.sn_convolution_2d import SNConvolution2D

def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)

def upsample_conv(x, conv):
    return conv(_upsample(x))

class GeneratorBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False):
        super(GeneratorBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(np.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.b1 = L.BatchNormalization(in_channels)
            self.b2 = L.BatchNormalization(hidden_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x, z=None, **kwargs):
        h = x
        h = self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, z=None, **kwargs):
        return self.residual(x, z, **kwargs) + self.shortcut(x)
    
class DiscriminatorBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False,sn=True):
        super(DiscriminatorBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(np.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            if sn:
                self.c1 = SNConvolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c2 = SNConvolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
                if self.learnable_sc:
                    self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
            else:
                self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
                if self.learnable_sc:
                    self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu,sn=True):
        super(OptimizedBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(np.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            if sn:
                self.c1 = SNConvolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c2 = SNConvolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
            else:
                self.c1 = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c2 = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
                
    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class Generator32(chainer.Chain):
    def __init__(self, ch=64, n_hidden=128, bottom_width=4, activation=F.relu,name='g'):
        super(Generator32, self).__init__()
        self.name = name
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.n_hidden = n_hidden
        self.fix_z = self.xp.random.normal(size=(100, n_hidden, 1, 1)).astype(np.float32)
        self.fix_z /= self.xp.sqrt(self.xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/n_hidden + 1e-8)
        with self.init_scope():
            
            self.l1 = L.Linear(n_hidden, (bottom_width ** 2) * ch * 2, initialW=initializer)
            self.block1 = GeneratorBlock(ch * 2, ch * 2, activation=activation, upsample=True)
            self.block2 = GeneratorBlock(ch * 2, ch * 2, activation=activation, upsample=True)
            self.block3 = GeneratorBlock(ch * 2, ch * 2, activation=activation, upsample=True)
            self.b4 = L.BatchNormalization(ch*2)
            self.l4 = L.Convolution2D(ch*2, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z
    
    def __call__(self, z):
        
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.b4(h)
        h = self.activation(h)
        h = self.l4(h)

        return h

class Discriminator32(chainer.Chain):
    def __init__(self, ch=64, activation=F.relu,sn=True):
        super(Discriminator32, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch*2,sn=sn)
            self.block2 = DiscriminatorBlock(ch * 2, ch * 2, activation=activation, downsample=True,sn=sn)
            self.block3 = DiscriminatorBlock(ch * 2, ch * 2, activation=activation, downsample=False,sn=sn)
            self.block4 = DiscriminatorBlock(ch * 2, ch * 2, activation=activation, downsample=False,sn=sn)
            if sn:
                self.l5 = SNLinear(ch * 2, 1, initialW=initializer)
            else:
                self.l5 = L.Linear(ch * 2, 1, initialW=initializer)

    def __call__(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l5(h)
        
        return output
    
class Generator48(chainer.Chain):
    def __init__(self, ch=64, dim_z=512, bottom_width=6, activation=F.relu,name='g'):
        super(Generator48, self).__init__()
        self.name = name
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.fix_z = self.xp.random.normal(size=(100, dim_z, 1, 1)).astype(np.float32)
        self.fix_z /= self.xp.sqrt(self.xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/dim_z + 1e-8)
        with self.init_scope():
            
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 8, initialW=initializer)
            self.block1 = GeneratorBlock(ch * 8, ch * 4, activation=activation, upsample=True)
            self.block2 = GeneratorBlock(ch * 4, ch * 2, activation=activation, upsample=True)
            self.block3 = GeneratorBlock(ch * 2, ch * 1, activation=activation, upsample=True)
            self.b4 = L.BatchNormalization(ch*1)
            self.l4 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.dim_z, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.dim_z + 1e-8)
        return z
    
    def __call__(self, z):
        
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.b4(h)
        h = self.activation(h)
        h = self.l4(h)

        return h

class Discriminator48(chainer.Chain):
    def __init__(self, ch=64, activation=F.relu):
        super(Discriminator48, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch * 1)
            self.block2 = DiscriminatorBlock(ch * 1, ch * 2, activation=activation, downsample=True)
            self.block3 = DiscriminatorBlock(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = DiscriminatorBlock(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = DiscriminatorBlock(ch * 8, ch * 16, activation=activation, downsample=False)
            self.l5 = SNLinear(ch * 16, 1, initialW=initializer)

    def __call__(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l5(h)
        
        return output

