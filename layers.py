import numpy as np
import theano
import theano.tensor as T

class OutputLayer(object):
	def __init__(self, input, n_in, n_out, activation="softmax", loss_type="nll"):
		self.n_in = n_in
		self.n_out = n_out

		self.W = theano.shared(
			value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
			name='W',
			borrow=True
			)
		
		self.b = theano.shared(
			value=np.zeros((n_out,), dtype=theano.config.floatX),
			name='b',
			borrow=True
			)

		self.output = T.dot(input, self.W) + self.b

		if activation == "softmax":
			self.p_y_given_x = T.nnet.softmax(self.output)
			self.y_pred = T.argmax(self.p_y_given_x, axis=1)
			self.output = self.p_y_given_x
			self.loss = lambda y: -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
			self.error = lambda y: T.mean(T.neq(self.y_pred, y))
		elif loss_type == "mse":
			self.loss = lambda y: T.mean(((y - self.output) ** 2).sum(axis=1))
			self.error = self.loss

		self.params = [self.W, self.b]

class Layer(object):
	def __init__(self, input, n_in, n_out, activation="relu"):
		self.n_in = n_in
		self.n_out = n_out

		self.W = theano.shared(
			value=np.asarray(np.random.uniform(
				low=-np.sqrt(6. / (n_in + n_out)),
				high=np.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)
			), dtype=theano.config.floatX),
			name='W',
			borrow=True
			)
		
		self.b = theano.shared(
			value=np.zeros((n_out,), dtype=theano.config.floatX),
			name='b',
			borrow=True
			)

		self.output = T.dot(input, self.W) + self.b

		if activation == "relu":
			act = lambda x : x * (x > 0)
		elif activation == "tanh":
			act = T.tanh
		elif activation == "sigmoid":
			act = T.nnet.sigmoid
		else:
			act = None
		
		self.output = self.output if activation is None else act(self.output)

		self.params = [self.W, self.b]

