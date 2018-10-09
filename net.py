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
        h = F.unpooling_2d(x, 2, 2, 0, outsize=(x.shape[2]*2, x.shape[3]*2))
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        return h
    
class DiscriminatorBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, initialW,sn=True):
        super(DiscriminatorBlock, self).__init__()
        with self.init_scope():
            if sn:
                self.c0 = SNConvolution2D(in_ch, in_ch, 3, 1, 1, initialW=initialW)
                self.c1 = SNConvolution2D(in_ch, out_ch, 3, 1, 1, initialW=initialW)
            else:
                self.c0 = L.Convolution2D(in_ch, in_ch, 3, 1, 1, initialW=initialW)
                self.c1 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=initialW)
    def __call__(self, x):
        h = F.leaky_relu((self.c0(x)))
        h = F.leaky_relu((self.c1(h)))
        h = F.average_pooling_2d(h, 2, 2, 0)
        return h

class Generator32(chainer.Chain):
    def __init__(self, n_hidden=512, ch=512,name='g'):
        super(Generator32, self).__init__()
        self.name = name
        self.n_hidden = n_hidden
        self.fix_z = self.xp.random.normal(size=(100, self.n_hidden, 1, 1)).astype(np.float32)
        self.fix_z /= self.xp.sqrt(self.xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.c0 = L.Convolution2D(n_hidden, ch, 4, 1, 3,initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1,initialW=w)
            self.bn1 = L.BatchNormalization(ch)

            self.b1 = GeneratorBlock(ch, ch//2)
            self.b2 = GeneratorBlock(ch//2, ch//4)
            self.b3 = GeneratorBlock(ch//4, ch//8)
            self.out3 = L.Convolution2D(ch//8, 3, 1, 1, 0,initialW=w)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z):

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        x = self.out3(h)

        return x

class Discriminator32(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02,sn=True):
        super(Discriminator32, self).__init__()
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            if sn:
                print('sn')
                self.in_ = SNConvolution2D(3, ch // 8, 1, 1, 0, initialW=w)
            else:
                print('sn_free')
                self.in_ = L.Convolution2D(3, ch // 8, 1, 1, 0, initialW=w)
                
            self.b3 = DiscriminatorBlock(ch // 8, ch // 4, initialW=w,sn=sn)
            self.b2 = DiscriminatorBlock(ch // 4, ch // 2, initialW=w,sn=sn)
            self.b1 = DiscriminatorBlock(ch // 2, ch, initialW=w,sn=sn)

            if sn:
                self.out0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
                self.out1 = SNConvolution2D(ch, ch, 4, 1, 0, initialW=w)
                self.out2 = SNLinear(ch, 1, initialW=w)
            else:
                self.out0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
                self.out1 = L.Convolution2D(ch, ch, 4, 1, 0, initialW=w)
                self.out2 = L.Linear(ch, 1, initialW=w) 
            
    def __call__(self, x):
        h = F.leaky_relu(self.in_(x))
        h = self.b3(h)
        h = self.b2(h)
        h = self.b1(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        h = self.out2(h)
        
        return h
    
class Generator48(chainer.Chain):
    def __init__(self, n_hidden=512, ch=512,name='g'):
        super(Generator48, self).__init__()
        self.name = name
        self.n_hidden = n_hidden
        self.fix_z = self.xp.random.normal(size=(100, self.n_hidden, 1, 1)).astype(np.float32)
        self.fix_z /= self.xp.sqrt(self.xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.c0 = L.Convolution2D(n_hidden, ch, 6, 1, 5,initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1,initialW=w)
            self.bn1 = L.BatchNormalization(ch)

            self.b1 = GeneratorBlock(ch, ch//2)
            self.b2 = GeneratorBlock(ch//2, ch//4)
            self.b3 = GeneratorBlock(ch//4, ch//8)
            self.out3 = L.Convolution2D(ch//8, 3, 1, 1, 0,initialW=w)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z):

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        x = self.out3(h)

        return x

class Discriminator48(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        super(Discriminator48, self).__init__()
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            self.in_ = SNConvolution2D(3, ch // 8, 1, 1, 0, initialW=w)
            self.b3 = DiscriminatorBlock(ch // 8, ch // 4, initialW=w)
            self.b2 = DiscriminatorBlock(ch // 4, ch // 2, initialW=w)
            self.b1 = DiscriminatorBlock(ch // 2, ch, initialW=w)

            self.out0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.out1 = SNConvolution2D(ch, ch, 6, 1, 0, initialW=w)
            self.out2 = SNLinear(ch, 1, initialW=w)
            
    def __call__(self, x):
        h = F.leaky_relu(self.in_(x))
        h = self.b3(h)
        h = self.b2(h)
        h = self.b1(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        h = self.out2(h)
        
        return h
    
class Generator64(chainer.Chain):
    def __init__(self, n_hidden=512, ch=512,name='g'):
        super(Generator64, self).__init__()
        self.name = name
        self.n_hidden = n_hidden
        self.fix_z = self.xp.random.normal(size=(100, self.n_hidden, 1, 1)).astype(np.float32)
        self.fix_z /= self.xp.sqrt(self.xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.c0 = L.Convolution2D(n_hidden, ch, 4, 1, 3,initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1,initialW=w)
            self.bn1 = L.BatchNormalization(ch)

            self.b0 = GeneratorBlock(ch, ch)
            self.b1 = GeneratorBlock(ch, ch//2)
            self.b2 = GeneratorBlock(ch//2, ch//4)
            self.b3 = GeneratorBlock(ch//4, ch//8)
            self.out3 = L.Convolution2D(ch//8, 3, 1, 1, 0,initialW=w)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z):

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = self.b0(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        x = self.out3(h)

        return x

class Discriminator64(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        super(Discriminator64, self).__init__()
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            self.in_ = SNConvolution2D(3, ch // 8, 1, 1, 0, initialW=w)
            self.b3 = DiscriminatorBlock(ch // 8, ch // 4, initialW=w)
            self.b2 = DiscriminatorBlock(ch // 4, ch // 2, initialW=w)
            self.b1 = DiscriminatorBlock(ch // 2, ch, initialW=w)
            self.b0 = DiscriminatorBlock(ch, ch, initialW=w)

            self.out0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.out1 = SNConvolution2D(ch, ch, 4, 1, 0, initialW=w)
            self.out2 = SNLinear(ch, 1, initialW=w)
            
    def __call__(self, x):
        h = F.leaky_relu(self.in_(x))
        h = self.b3(h)
        h = self.b2(h)
        h = self.b1(h)
        h = self.b0(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        h = self.out2(h)
        
        return h

class Generator128(chainer.Chain):
    def __init__(self, n_hidden=512, ch=512,name='g'):
        super(Generator128, self).__init__()
        self.name = name
        self.n_hidden = n_hidden
        self.fix_z = self.xp.random.normal(size=(100, self.n_hidden, 1, 1)).astype(np.float32)
        self.fix_z /= self.xp.sqrt(self.xp.sum(self.fix_z*self.fix_z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.c0 = L.Convolution2D(n_hidden, ch, 4, 1, 3,initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1,initialW=w)
            self.bn1 = L.BatchNormalization(ch)

            self.b0 = GeneratorBlock(ch, ch)
            self.b1 = GeneratorBlock(ch, ch)
            self.b2 = GeneratorBlock(ch, ch//2)
            self.b3 = GeneratorBlock(ch//2, ch//4)
            self.b4 = GeneratorBlock(ch//4, ch//8)
            self.out = L.Convolution2D(ch//8, 3, 1, 1, 0,initialW=w)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z):

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = self.b0(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        x = self.out(h)

        return x
    
class Discriminator128(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        super(Discriminator128, self).__init__()
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            self.in_ = SNConvolution2D(3, ch // 8, 1, 1, 0, initialW=w)
            self.b4 = DiscriminatorBlock(ch // 8, ch // 4, initialW=w)
            self.b3 = DiscriminatorBlock(ch // 4, ch // 2, initialW=w)
            self.b2 = DiscriminatorBlock(ch // 2, ch, initialW=w)
            self.b1 = DiscriminatorBlock(ch, ch, initialW=w)
            self.b0 = DiscriminatorBlock(ch, ch, initialW=w)

            self.out0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.out1 = SNConvolution2D(ch, ch, 4, 1, 0, initialW=w)
            self.out2 = SNLinear(ch, 1, initialW=w)
            
    def __call__(self, x):
        h = F.leaky_relu(self.in_(x))
        h = self.b4(h)
        h = self.b3(h)
        h = self.b2(h)
        h = self.b1(h)
        h = self.b0(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        h = self.out2(h)
        
        return h