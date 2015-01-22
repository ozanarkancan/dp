import theano
import theano.tensor as T
import numpy as np, h5py
import argparse
from layers import *
from optim import *
import time

def get_arg_parser():
	parser = argparse.ArgumentParser(prog="experiment")
	parser.add_argument("--data_path", default="/home/can/data/dp/archybrid_conllWSJToken_wikipedia2MUNK-100_fv021a_xy.mat", help="data path")
	parser.add_argument("--minibatch", default=128, type=int, help="minibatch size")
	parser.add_argument("--opt", default="sgd", help="optimization type sgd, rmsprop, adagrad")
	parser.add_argument("--reg", default="None", help="regularization l1, l2, dropout")
	parser.add_argument("--hidden", default=[20000], nargs='+', type=int, help="number of units in hidden layer(s)")
	parser.add_argument("--epoch", default=1, type=int, help="number of epochs")

	return parser

def shared_dataset(data_xy):
	data_x, data_y = data_xy
		
	shared_x = theano.shared(
		np.asarray(data_x, dtype=theano.config.floatX),
			borrow=True
			)

	shared_y = theano.shared(
		np.asarray(data_y.reshape(data_y.shape[0], ), dtype=theano.config.floatX),
			borrow=True
			)

	return shared_x, T.cast(shared_y, 'int32')

if __name__ == "__main__":
	parser = get_arg_parser()

	args = vars(parser.parse_args())
	data_path = args['data_path']
	ms = args['minibatch']
	opt = args['opt']
	reg = args['reg']
	hidden = args['hidden']
	epochs = args['epoch']

	print "...loading data"
	f = h5py.File(data_path, 'r')
	x_trn = f.get('x_trn')
	x_trn = np.array(x_trn)
	y_trn = f.get('y_trn')
	y_trn = np.array(y_trn) - 1
	x_dev = f.get('x_dev')
	x_dev = np.array(x_dev)
	y_dev = f.get('y_dev')
	y_dev = np.array(y_dev) - 1

	num_classes = len(np.unique(y_dev))
	
	x = T.matrix('x')
	y = T.ivector('y')
	
	layers = []

	layer0 = Layer(x, x_trn.shape[1], hidden[0])
	layers.append(layer0)

	for i in range(len(hidden) - 1):
		layer = Layer(layers[-1].output, layers[-1].n_out, hidden[i + 1])
		layers.append(layer)
	
	layer_out = OutputLayer(layers[-1].output, layers[-1].n_out, num_classes)
	layers.append(layer_out)

	params = []
	for l in layers:
		print (l.n_in, l.n_out)
		params += l.params

	cost = layers[-1].loss(y)
	errors = layers[-1].error

	#train_x, train_y = shared_dataset([
	#	np.asarray(np.random.random((ms, x_trn.shape[1])), dtype='float32'),
	#	np.asarray(np.random.randint(num_classes, ms, 1), dtype='int32')
	#])

	#valid_x, valid_y = shared_dataset([
	#	np.asarray(x_dev, dtype='float32'),
	#	np.asarray(y_dev, dtype='int32')
	#])

	print "...building the model"

	acc_err = theano.function(
		inputs = [x, y],
		outputs = errors(y),
		)
	
	loss_err = theano.function(
		inputs = [x, y],
		outputs = cost
		)
	
	both_err = theano.function(
		inputs = [x, y],
		outputs = [errors(y), cost]
		)
	
	grads = T.grad(cost, params)

	if opt == "rmsprop":
		updates = rmsprop(params, grads)
	else:
		updates = sgd(params, grads)

	train_model = theano.function(
		inputs = [x, y],
		outputs = cost,
		updates = updates
		)
	
	n_train_batches = x_trn.shape[0] / ms
	n_valid_batches = x_dev.shape[0] / ms

	print "...training the model"

	for i in xrange(epochs):
		start = time.time()
		costs = [train_model(x_trn[index * ms : (index + 1) * ms, :],\
				y_trn[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_train_batches)]
		
		avg_trn_loss = np.mean(costs)

		trn_acc_errs = [acc_err(x_trn[index * ms : (index + 1) * ms, :],\
				y_trn[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_train_batches)]
		avg_trn_err = np.mean(trn_acc_errs)

		#dev_losses = [loss_err(x_dev[index * ms : (index + 1) * ms, :],\
		#		y_dev[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_valid_batches)]
		#avg_dev_loss = np.mean(dev_losses)
	
		errs = [both_err(x_dev[index * ms : (index + 1) * ms, :],\
				y_dev[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_valid_batches)]

		dev_acc_errs = []
		dev_losses = []

		for e in errs:
			dev_acc_errs.append(e[0])
			dev_losses.append(e[1])

		avg_dev_err = np.mean(dev_acc_errs)
		avg_dev_loss = np.mean(dev_losses)
		
		end = time.time()
		print "Epoch: %i\ntrain loss: %f\ntrain err: %f\ndev loss: %f\ndev error: %f\nRunning Time: %f seconds\n" % (i + 1, avg_trn_loss, avg_trn_err, avg_dev_loss, avg_dev_err, end - start)
	
