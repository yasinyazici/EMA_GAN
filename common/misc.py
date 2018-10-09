# from https://github.com/chainer/chainerrl/blob/f119a1fe210dd31ea123d244258d9b5edc21fba4/chainerrl/misc/copy_param.py

from chainer import links as L
import chainer


def copy_param(target_link, source_link):
    """Copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data

    # Copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] = link.avg_mean
            target_bn.avg_var[:] = link.avg_var


def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] *= (1 - tau)
        target_params[param_name].data[:] += tau * param.data

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] *= (1 - tau)
            target_bn.avg_mean[:] += tau * link.avg_mean
            target_bn.avg_var[:] *= (1 - tau)
            target_bn.avg_var[:] += tau * link.avg_var
            
def soft_copy_param_init(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        if param_name in ['/c0/b','/c0/W','/bn0/beta','/bn0/gamma']:
            target_params[param_name].data[:] *= (1 - tau)
            target_params[param_name].data[:] += tau * param.data
        else:
            target_params[param_name].data[:] = param.data

    # Copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            if param_name in ['/bn0']:
                target_bn = target_links[link_name]
                target_bn.avg_mean[:] *= (1 - tau)
                target_bn.avg_mean[:] += tau * link.avg_mean
                target_bn.avg_var[:] *= (1 - tau)
                target_bn.avg_var[:] += tau * link.avg_var
            else:
                target_bn = target_links[link_name]
                target_bn.avg_mean[:] = link.avg_mean
                target_bn.avg_var[:] = link.avg_var
            
def average_param(target_link, source_link, n_model):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] *= (1.0*n_model/(n_model+1))
        target_params[param_name].data[:] += (1.0/(n_model+1)) * param.data

    # average Batch Normalization's statistics (Should we stick with EMA for BacthNorm?)
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            #target_bn.avg_mean[:] *= (1 - tau)
            #target_bn.avg_mean[:] += tau * link.avg_mean
            #target_bn.avg_var[:] *= (1 - tau)
            #target_bn.avg_var[:] += tau * link.avg_var
            target_bn.avg_mean[:] *= (1.0*n_model/(n_model+1))
            target_bn.avg_mean[:] += (1.0/(n_model+1)) * link.avg_mean
            target_bn.avg_var[:] *= (1.0*n_model/(n_model+1))
            target_bn.avg_var[:] += (1.0/(n_model+1)) * link.avg_var
            
def inc_batch(mul=2):
    @chainer.training.make_extension()
    def increase(trainer):
        trainer.updater.get_iterator('main').batchsize *=mul
        
    return increase