import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

def dropout(input, srng, dropout_rate):
	mask = srng.binomial(n=1, p=(1 - dropout_rate), size=input.shape)
	d_output = input * T.cast(mask, theano.config.floatX)
	return d_output

def initialize_weights(n_in, n_out):
	W = theano.shared(
		value=np.asarray(np.random.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
		), dtype=theano.config.floatX),
		name='W',
		borrow=True
		)
	
	b = theano.shared(
		value=np.zeros((n_out,), dtype=theano.config.floatX),
		name='b',
		borrow=True
		)

	return W, b
	

class InputLayer(object):
	def __init__(self, input, srng, dropout_rate=0.5):
		self.output = input * (1 - dropout_rate)
		self.d_output = dropout(input, srng, dropout_rate)

class OutputLayer(object):
	def __init__(self, input, d_input, n_in, n_out, activation="softmax", loss_type="nll"):
		self.n_in = n_in
		self.n_out = n_out

		self.W, self.b = initialize_weights(n_in, n_out)

		self.output = T.dot(input, self.W) + self.b
		self.d_output = T.dot(d_input, self.W) + self.b

		if activation == "softmax":
			self.p_y_given_x = T.nnet.softmax(self.output)
			self.y_pred = T.argmax(self.p_y_given_x, axis=1)
			self.output = self.p_y_given_x
			self.loss = lambda y: -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
			self.error = lambda y: T.mean(T.neq(self.y_pred, y))

			self.d_p_y_given_x = T.nnet.softmax(self.d_output)
			self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
			self.d_output = self.d_p_y_given_x
			self.d_loss = lambda y: -T.mean(T.log(self.d_p_y_given_x)[T.arange(y.shape[0]), y])
			self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))
		elif loss_type == "mse":
			self.loss = lambda y: T.mean(((y - self.output) ** 2).sum(axis=1))
			self.error = self.loss

			self.d_loss = lambda y: T.mean(((y - self.d_output) ** 2).sum(axis=1))
			self.d_error = self.d_loss

		self.params = [self.W, self.b]

class Layer(object):
	def __init__(self, input, d_input, n_in, n_out, srng, dropout_rate=0.5, activation="relu"):
		self.n_in = n_in
		self.n_out = n_out

		self.W, self.b = initialize_weights(n_in, n_out)

		self.output = T.dot(input, self.W) + self.b
		self.d_output = T.dot(d_input, self.W) + self.b

		if activation == "relu":
			act = lambda x : x * (x > 0)
		elif activation == "tanh":
			act = T.tanh
		elif activation == "sigmoid":
			act = T.nnet.sigmoid
		else:
			act = None
		
		self.output = self.output if activation is None else act(self.output)
		self.d_output = self.d_output if activation is None else act(self.d_output)

		self.d_output = dropout(self.d_output, srng, dropout_rate)
		self.output = self.output * (1 - dropout_rate)

		self.params = [self.W, self.b]
