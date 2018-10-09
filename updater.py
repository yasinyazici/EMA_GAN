import numpy as np
import os, sys

import chainer
import chainer.functions as F
from chainer import Variable

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)
from common.misc import soft_copy_param,copy_param,average_param

# Classic Adversarial Loss

def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss

def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss

def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss

# WGAN loss
def loss_wgan_dis(dis_fake, dis_real):
    L1 = F.sum(-dis_real)
    L2 = F.sum(dis_fake)
    loss = L1 + L2
    return loss

def loss_wgan_gen(dis_fake):
    loss = F.sum(-dis_fake)
    return loss

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.g_ema,self.g_ma = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.smoothing = kwargs.pop('smoothing')
        self.objective = kwargs.pop('objective')
        self.ma_start = kwargs.pop('ma_start')
        self.counter = 0
        self.n_model = 0
        self.lam =10.0
        if self.objective == 'gan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.objective == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        elif self.objective == 'wgan-gp':
            self.loss_gen = loss_wgan_gen
            self.loss_dis = loss_wgan_dis
        else:
            NotImplementedError('no such objective')
            
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp
        
        self.counter += 1

        for i in range(self.n_dis):

            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x = []
            for j in range(batchsize):
                x.append(np.asarray(batch[j]).astype("f"))
            x_real = Variable(xp.asarray(x))
            
            if i == 0:
                # train generator
                z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
                x_fake = self.gen(z)
                y_fake = self.dis(x_fake)
                loss_gen = self.loss_gen(y_fake)#F.sum(F.softplus(-y_fake)) / batchsize
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})
                soft_copy_param(self.g_ema, self.gen, 1.0-self.smoothing)
                if self.counter >= self.ma_start:
                    average_param(self.g_ma,self.gen,self.n_model)
                    self.n_model += 1


            y_real = self.dis(x_real)

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)

            x_fake.unchain_backward()

            loss_dis = self.loss_dis(y_fake,y_real)
            
            if self.objective =='wgan-gp':
                eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
                x_mid = eps * x_real + (1.0 - eps) * x_fake

                x_mid_v = Variable(x_mid.data)
                y_mid = F.sum(self.dis(x_mid_v))

                dydx, = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
                dydx = F.sqrt(F.sum((dydx*dydx), axis=(1, 2, 3)))                
                loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))
                loss_dis += loss_gp
                chainer.reporter.report({'loss_gp': loss_gp})
            
            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            loss_dis.unchain_backward()

            chainer.reporter.report({'loss_dis': loss_dis})
