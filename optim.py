import theano
import theano.tensor as T

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

#def adagrad(params, grads, )
