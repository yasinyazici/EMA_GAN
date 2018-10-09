import sys
import os
import math


import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
#from sn.sn_linear import SNLinear
#from sn.sn_convolution_2d import SNConvolution2D
from common.sn.sn_linear import SNLinear
from common.sn.sn_convolution_2d import SNConvolution2D

def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if not chainer.config.train:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)

def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, 2, 0, outsize=(h * 2, w * 2))

def upsample_conv(x, conv):
    return conv(_upsample(x))

def _rms(x):
    return F.sqrt(F.mean((F.square(x))))

# differentiable backward functions
def backward_linear(x_in, x, l):
    y = F.matmul(x, l.W)
    return y

def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))
    return y

def backward_deconvolution(x_in, x, l):
    y = F.convolution_2d(x, l.W, None, l.stride, l.pad)
    return y

def backward_relu(x_in, x):
    y = (x_in.data > 0) * x
    return y


def backward_leaky_relu(x_in, x, a):
    y = (x_in.data > 0) * x + a * (x_in.data < 0) * x
    return y


def backward_sigmoid(x_in, g):
    y = F.sigmoid(x_in)
    return g * y * (1 - y)

class SNDiscriminatorBlock(chainer.Chain):
    # conv-conv-downsample
    def __init__(self, in_ch, out_ch,initialW):
        super(SNDiscriminatorBlock, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(in_ch, in_ch, 3, 1, 1,initialW=initialW)
            self.c1 = SNConvolution2D(in_ch, out_ch, 3, 1, 1,initialW=initialW)
    def __call__(self, x):
        h = F.leaky_relu((self.c0(x))) 
        h = F.leaky_relu((self.c1(h)))
        h = F.average_pooling_2d(h, 2, 2, 0)
        return h#, _rms(h)

class GeneratorBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        super(GeneratorBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1,initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1,initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)
    def __call__(self, x):
        h = _upsample(x)
        h = F.leaky_relu(self.bn0(self.c0(h))) 
        h = F.leaky_relu(self.bn1(self.c1(h)))
        return h  

class BlockDisc(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(BlockDisc, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = SNConvolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = SNConvolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

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

    def __call__(self, x,with_norm=False):
        res = self.residual(x)
        short = self.shortcut(x)
        if with_norm:
            return res + short, _rms(res)/_rms(short)
        else:
            return res + short
    
class BlockGen(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False):
        super(BlockGen, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
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

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
    
class OptimizedBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = SNConvolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = SNConvolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

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

class ResBlock(chainer.Chain):

    def __init__(self,ch, wscale=0.02):
        super(ResBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, 3, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(3)
            self.bn1 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = self.c0(F.relu(self.bn0(x)))
        hs = self.c1(F.relu(self.bn1(h)))
        return x + hs
    
class ResFCBlock(chainer.Chain):

    def __init__(self,ch):
        super(ResFCBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.l0 = L.Linear(ch, ch,initialW=initializer)
            self.l1 = L.Linear(ch, ch,initialW=initializer)
            self.bn0 = L.BatchNormalization(ch)
            self.bn1 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = self.l0(F.relu(self.bn0(x)))
        hs = self.l1(F.relu(self.bn1(h)))
        return x + hs
    
class SNResFCBlock(chainer.Chain):

    def __init__(self,ch):
        super(SNResFCBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.l0 = SNLinear(ch, ch,initialW=initializer)
            self.l1 = SNLinear(ch, ch,initialW=initializer)
            self.bn0 = L.BatchNormalization(ch)
            self.bn1 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = self.l0(F.relu(self.bn0(x)))
        hs = self.l1(F.relu(self.bn1(h)))
        return x + hs
    
class UpResBlock(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch, wscale=0.02):
        super(UpResBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.bn1 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = self.c0(F.unpooling_2d(F.relu(self.bn0(x)), 2, 2, 0, cover_all=False))
        h = self.c1(F.relu(self.bn1(h)))
        hs = self.cs(F.unpooling_2d(x, 2, 2, 0, cover_all=False))
        return h + hs

# common generators

class DCGANGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=512, wscale=0.02,
                 z_distribution="uniform", hidden_activation=F.relu, output_activation=F.tanh, use_bn=True):
        super(DCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x
    
class PGGANGenerator(chainer.Chain):
    def __init__(self, ch=512, dim_z=512):
        super(PGGANGenerator, self).__init__()
        self.dim_z = dim_z
        xp = self.xp
        self.fix_z = xp.random.normal(size=(100, dim_z, 1, 1)).astype(np.float32)
        self.fix_z /= xp.sqrt(xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/dim_z + 1e-8)
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.c0 = L.Convolution2D(dim_z, ch, 4, 1, 3, initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.bn1 = L.BatchNormalization(ch)
            self.b1 = GeneratorBlock(ch, ch//2)
            self.b2 = GeneratorBlock(ch//2, ch//4)
            self.b3 = GeneratorBlock(ch//4, ch//8)
            self.out = L.Convolution2D(ch//8, 3, ksize=3, stride=1, pad=1, initialW=w)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.dim_z, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.dim_z + 1e-8)
        return z

    def __call__(self, z):
        h = F.reshape(z,(len(z), self.dim_z, 1, 1))
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.out(h)

        return h

class ResNetGenerator(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.fix_z = self.make_hidden(100)
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 8, initialW=initializer)
            self.block2 = BlockGen(ch * 8, ch * 4, activation=activation, upsample=True)
            self.block3 = BlockGen(ch * 4, ch * 2, activation=activation, upsample=True)
            self.block4 = BlockGen(ch * 2, ch, activation=activation, upsample=True)
            self.b5 = L.BatchNormalization(ch)
            self.l5 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def make_hidden(self, batchsize):
        if self.distribution == "normal":
            return np.random.randn(batchsize, self.dim_z, 1, 1).astype(np.float32)
        elif self.distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.dim_z, 1, 1)).astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.distribution)
            
    def __call__(self, z):
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = F.tanh(self.l5(h))
        return h
    
'''    
class ResnetGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, z_distribution="normal", wscale=0.02):
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        super(ResnetGenerator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(n_hidden, n_hidden * bottom_width * bottom_width)
            self.r0 = UpResBlock(n_hidden)
            self.r1 = UpResBlock(n_hidden)
            self.r2 = UpResBlock(n_hidden)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.c3 = L.Convolution2D(n_hidden, 3, 3, 1, 1, initialW=w)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, x):
        h = F.reshape(F.relu(self.l0(x)), (x.data.shape[0], self.n_hidden, self.bottom_width, self.bottom_width))
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        h = self.bn2(F.relu(h))
        h = F.tanh(self.c3(h))
        return h
'''

# common discriminators

class SNResNetDiscriminator(chainer.Chain):
    def __init__(self, ch=64, activation=F.relu):
        super(SNResNetDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        #w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            #self.block1 = L.Convolution2D(3, ch // 8, 1, 1, 0, initialW=w)
            self.block2 = BlockDisc(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = BlockDisc(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = BlockDisc(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = BlockDisc(ch * 8, ch * 8, activation=activation, downsample=False)
            self.l6 = SNLinear(ch * 8, 1, initialW=initializer)

    def __call__(self, x):
        h = x
        h = self.block1(h)
        h,h_n1 = self.block2(h,with_norm=True)
        h,h_n2 = self.block3(h,with_norm=True)
        h,h_n3 = self.block4(h,with_norm=True)
        h,h_n4 = self.block5(h,with_norm=True)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l6(h)
        return output, F.stack([h_n1,h_n2,h_n3,h_n4],axis=0)
    
class SNPGGANDiscriminator(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(SNPGGANDiscriminator, self).__init__()
        with self.init_scope():
            self.in1 = SNConvolution2D(3, ch // 8, 1, 1, 0, initialW=w)
            self.b1 = SNDiscriminatorBlock(ch // 8, ch // 4, initialW=w)
            self.b2 = SNDiscriminatorBlock(ch // 4, ch // 2, initialW=w)
            self.b3 = SNDiscriminatorBlock(ch // 2, ch, initialW=w)
            self.out0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.out1 = SNConvolution2D(ch, ch, 4, 1, 0, initialW=w)
            self.out2 = SNLinear(ch, 1, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.in1(x))
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = F.leaky_relu(self.out0(h))
        h = F.leaky_relu(self.out1(h))
        return self.out2(h), None# F.stack([h_n1,h_n2,h_n3],axis=0)
    
class DCGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(DCGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        return self.l4(h)


class SNDCGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SNDCGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = SNConvolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = SNConvolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = SNConvolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = SNConvolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = SNConvolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = SNConvolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = SNConvolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c2_1(h))
        h = F.leaky_relu(self.c3_0(h))
        return self.l4(h)



class WGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(WGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h0 = F.leaky_relu(self.c0(self.x))
        self.h1 = F.leaky_relu(self.c1(self.h0))
        self.h2 = F.leaky_relu(self.c1_0(self.h1))
        self.h3 = F.leaky_relu(self.c2(self.h2))
        self.h4 = F.leaky_relu(self.c2_0(self.h3))
        self.h5 = F.leaky_relu(self.c3(self.h4))
        self.h6 = F.leaky_relu(self.c3_0(self.h5))
        return self.l4(self.h6)

    def differentiable_backward(self, x):
        g = backward_linear(self.h6, x, self.l4)
        g = F.reshape(g, (x.shape[0], 512, 4, 4))
        g = backward_leaky_relu(self.h6, g, 0.2)
        g = backward_convolution(self.h5, g, self.c3_0)
        g = backward_leaky_relu(self.h5, g, 0.2)
        g = backward_convolution(self.h4, g, self.c3)
        g = backward_leaky_relu(self.h4, g, 0.2)
        g = backward_convolution(self.h3, g, self.c2_0)
        g = backward_leaky_relu(self.h3, g, 0.2)
        g = backward_convolution(self.h2, g, self.c2)
        g = backward_leaky_relu(self.h2, g, 0.2)
        g = backward_convolution(self.h1, g, self.c1_0)
        g = backward_leaky_relu(self.h1, g, 0.2)
        g = backward_convolution(self.h0, g, self.c1)
        g = backward_leaky_relu(self.h0, g, 0.2)
        g = backward_convolution(self.x, g, self.c0)
        return g


class DownResBlock1(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(3, ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0((self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4

    def differentiable_backward(self, g):
        gs = backward_convolution(self.h0, g, self.cs)
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        # g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g


class DownResBlock2(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4

    def differentiable_backward(self, g):
        gs = backward_convolution(self.h0, g, self.cs)
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g


class DownResBlock3(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        return self.h4

    def differentiable_backward(self, g):
        gs = g
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g


class ResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(ResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = DownResBlock1(128)
            self.r1 = DownResBlock2(128)
            self.r2 = DownResBlock3(128)
            self.r3 = DownResBlock3(128)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        return self.l4(F.relu(self.h4))

    def differentiable_backward(self, x):
        g = backward_linear(self.h4, x, self.l4)
        g = F.reshape(g, (x.shape[0], self.ch, self.bottom_width, self.bottom_width))
        g = backward_leaky_relu(self.h4, g, 0.0)
        g = self.r3.differentiable_backward(g)
        g = self.r2.differentiable_backward(g)
        g = self.r1.differentiable_backward(g)
        g = self.r0.differentiable_backward(g)
        return g
