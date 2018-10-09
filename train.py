import argparse
import os
import sys

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.dataset import Cifar10Dataset,Stl10Dataset,Stl10_48_Dataset,Imagenet32Dataset,Imagenet64Dataset,CelebADataset
from evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID
from common.record import record_setting

from net import Discriminator32, Generator32, Discriminator48, Generator48,\
                Discriminator64, Generator64, Discriminator128, Generator128
from resnet import Discriminator32 as Discriminator32_resnet
from resnet import Generator32 as Generator32_resnet
from updater import Updater

#from common.OptAdam import OptAdam

from common.misc import copy_param

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--dataset', '-d',type=str, default="cifar10")
    parser.add_argument('--size', '-s',type=int, default=32)
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--objective', '-o', type=str, default='gan') #gan, hinge, wgan-gp
    parser.add_argument('--n_dis', type=int, default=1,
                        help='number of discriminator update per generator update')
    parser.add_argument('--max_iter', '-m', type=int, default=500000)
    parser.add_argument('--model', type=str, default='base') #base, resnet
    #parser.add_argument('--type_of_averaging', type=str, default='ema') # ma,ema
    parser.add_argument('--generator_smoothing', type=float, default=0.9999)
    parser.add_argument('--ma_start', type=float, default=50000)
    #parser.add_argument('--sn', type=bool, default=False)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    #parser.add_argument('--snapshot_interval', type=int, default=500000,
    #                    help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=20000,
                        help='Interval of evaluation')
    parser.add_argument('--out_image_interval', type=int, default=1000,
                        help='Interval of evaluation')
    parser.add_argument('--stage_interval', type=int, default=400000,
                        help='Interval of stage progress')
    parser.add_argument('--display_interval', type=int, default=1000,
                        help='Interval of displaying log to console')
    #parser.add_argument('--pretrained_generator', type=str, default="")
    #parser.add_argument('--pretrained_discriminator', type=str, default="")

    args = parser.parse_args()
    #record_setting(args.out)

    if args.objective == 'wgan-gp':
        report_keys = ["loss_dis", "loss_gen", "loss_gp", "IS_ema", "FID_ema", "IS_ma", "FID_ma","IS","FID"]
    else:
        report_keys = ["loss_dis", "loss_gen", "IS_ema", "FID_ema", "IS_ma", "FID_ma","IS","FID"]
    
    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    if args.size == 32 and args.model =='base':
        Generator = Generator32
        Discriminator = Discriminator32
    elif args.size == 32 and args.model =='resnet':
        Generator = Generator32_resnet
        Discriminator = Discriminator32_resnet
    elif args.size == 48 and args.model =='base':
        Generator = Generator48
        Discriminator = Discriminator48
    elif args.size == 64 and args.model =='base':
        Generator = Generator64
        Discriminator = Discriminator64
    elif args.size == 128 and args.model =='base':
        Generator = Generator128
        Discriminator = Discriminator128
    else:
        NotImplementedError('no such model or size')

    sn = True
    if args.objective in ['wgan-gp']:
        sn = False
        
	print(sn)
    
    n_hidden = 512   
    ch = 512
    if args.model =='resnet':
        n_hidden = 128
        if sn:
            ch = 128 
        else: 
            ch =64
        
    generator = Generator(n_hidden=n_hidden,ch=ch)#decay=0.9)
    generator_ema = Generator(n_hidden=n_hidden,ch=ch,name='g_ema')
    generator_ma = Generator(n_hidden=n_hidden,ch=ch,name='g_ma')
    discriminator = Discriminator(ch=ch,sn=sn)

    # select GPU
    if args.gpu >= 0:
        generator.to_gpu()
        generator_ema.to_gpu()
        generator_ma.to_gpu()
        discriminator.to_gpu()
        print("use gpu {}".format(args.gpu))

    #if args.pretrained_generator != "":
    #    chainer.serializers.load_npz(args.pretrained_generator, generator)
    #if args.pretrained_discriminator != "":
    #    chainer.serializers.load_npz(args.pretrained_discriminator, discriminator)
    copy_param(generator_ema, generator)
    copy_param(generator_ma, generator)

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.0, beta2=0.9):
        #optimizer = OptAdam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer

    opt_gen = make_optimizer(generator)
    opt_dis = make_optimizer(discriminator)
    
    if args.dataset == 'cifar10':
        train_dataset = Cifar10Dataset()
    elif args.dataset == 'stl10'and args.size == 48:
        train_dataset = Stl10_48_Dataset()
    elif args.dataset == 'stl10':
        train_dataset = Stl10Dataset(resize=args.size)
    elif args.dataset == 'imagenet' and args.size == 32:
        train_dataset = Imagenet32Dataset()
    elif args.dataset == 'imagenet' and args.size == 64:
        train_dataset = Imagenet64Dataset()
    elif args.dataset == 'celeba':
        train_dataset = CelebADataset(resize=args.size)
    else:
        NotImplementedError('no such dataset')
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

    fix_z = generator.fix_z
    im_gen_pre_fix = args.model +'_'+ args.objective +'_' +str(args.n_dis) +'_' + args.dataset+'_'+str(args.size)

    # Set up a trainer
    updater = Updater(
        models=(generator, discriminator, generator_ema, generator_ma),
        iterator={
            'main': train_iter},
        optimizer={
            'opt_gen': opt_gen,
            'opt_dis': opt_dis},
        device=args.gpu,
        n_dis=args.n_dis,
        smoothing=args.generator_smoothing,
        ma_start=args.ma_start,
        objective=args.objective
    )
    out_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + args.out)
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=out_path)
    
    trainer.extend(extensions.LogReport(keys=report_keys,trigger=(args.display_interval, 'iteration'),
                   log_name = im_gen_pre_fix))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    
    # for with smoothing
    trainer.extend(sample_generate(generator_ema, im_gen_pre_fix +'_w_ema',fix_z),
                   trigger=(args.out_image_interval, 'iteration'),priority=extension.PRIORITY_WRITER)
    trainer.extend(sample_generate(generator_ma, im_gen_pre_fix +'_w_ma',fix_z),
                   trigger=(args.out_image_interval, 'iteration'),priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_FID(generator_ema,args.dataset,args.size), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_FID(generator_ma,args.dataset,args.size), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    if args.dataset is not 'celeba': 
        trainer.extend(calc_inception(generator_ema), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
        trainer.extend(calc_inception(generator_ma), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
        
    # for w/o smoothing
    trainer.extend(sample_generate(generator, im_gen_pre_fix +'_wos',fix_z),
                   trigger=(args.out_image_interval, 'iteration'),priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_FID(generator,args.dataset,args.size), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    if args.dataset is not 'celeba':
        trainer.extend(calc_inception(generator), trigger=(args.evaluation_interval, 'iteration'),
                       priority=extension.PRIORITY_WRITER)
    
    trainer.extend(extensions.ProgressBar(update_interval=args.display_interval))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
