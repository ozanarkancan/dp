import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

def clip_norm(g, c, n):
	if c > 0:
		g = T.switch(T.ge(n, c), g*c/n, g)
	return g

def clip_norms(gs, c):
	norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
	return [clip_norm(g, c, norm) for g in gs]

def max_norm(p, maxnorm=0.):
	if maxnorm > 0:
		norms = T.sqrt(T.sum(T.sqr(p), axis=0))
		desired = T.clip(norms, 0, maxnorm)
		p = p * (desired/ (1e-7 + norms))
	return p

def gradient_regularize(p, g, l1=0., l2=0.):
	g += p * l2
	g += T.sgn(p) * l1
	return g

def weight_regularize(p, maxnorm=0.):
	p = max_norm(p, maxnorm)
	return p

def sgd(params, grads, lr=0.01):
	updates = [(p, p - lr * g) for p, g in zip(params, grads)]
	return updates

def rmsprop(params, grads, lr=0.001, rho=0.9, epsilon=1e-6):
	updates = []
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value() * 0.) # acc is allocated for each parameter (p) with 0 values with the shape of p
		acc_new = rho * acc + (1 - rho) * g ** 2
		gradient_scaling = T.sqrt(acc_new + epsilon)
		g = g / gradient_scaling
		updates.append((acc, acc_new))
		updates.append((p, p - lr * g))
	
	return updates

def adagrad(params, grads, lr=1.0, epsilon=1e-6):
	accs = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in params]
	
	updates = []
	for param_i, grad_i, acc_i in zip(params, grads, accs):
		acc_i_new = acc_i + grad_i**2
		updates.append((acc_i, acc_i_new))
		updates.append((param_i, param_i - lr * grad_i / T.sqrt(acc_i_new + epsilon)))

	return updates

def adam(params, grads, lr=0.0002, b1=0.1, b2=0.001, eps=1e-8):
	updates = OrderedDict()
	it = theano.shared(np.asarray(0., dtype=theano.config.floatX))
	updates[it] = it + 1.
	fix1 = 1. - (1. -b1) ** (it + 1.)
	fix2 = 1. - (1. - b2) ** (it + 1.)
	lr_t = lr * (T.sqrt(fix2) / fix1)
	for p, g in zip(params, grads):
		mom1 = theano.shared(np.asarray(p.get_value() * 0., dtype=theano.config.floatX))
		mom2 = theano.shared(np.asarray(p.get_value() * 0., dtype=theano.config.floatX))
		mom1_new = mom1 + b1 * (g - mom1)
		mom2_new = mom2 + b2 * (T.sqr(g) - mom2)

		effgrad = mom1_new / (T.sqrt(mom2_new) + eps)
		effstep_new = lr_t * effgrad

		w_new = p + effstep_new
		updates[p] = w_new
		updates[mom1] = mom1_new
		updates[mom2] = mom2_new
	
	return updates

