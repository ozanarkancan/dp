from layers import *
from optim import *

class DNET(object):
	def __init__(self, num_classes):
		self.layers = []
		self.rng = np.random.RandomState()
		self.srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
		self.num_classes = num_classes
		self.o_layer = None
	
	def add_input_layer(self, input, dropout_rate=0.5):
		input_layer = InputLayer(input, self.srng, dropout_rate=dropout_rate)
		self.layers.append(input_layer)
	
	def add_hidden_layer(self, n_in, n_out, dropout_rate=0.5, activation="relu"):
		layer = Layer(self.layers[-1].output, self.layers[-1].d_output, n_in, n_out, self.srng, dropout_rate, activation)
		self.layers.append(layer)
	
	def connect_output(self):
		self.o_layer = OutputLayer(self.layers[-1].output, self.layers[-1].d_output, self.layers[-1].n_out, self.num_classes)

	def get_params(self):
		params = []
		for l in self.layers:
			if not l.params == None:
				params += l.params

		params += self.o_layer.params
		return params
	
	def save(self):
		filename = os.path.realpath('.') + "/dp/save/net.save"
		f = file(filename, 'wb')
		for p in params:
			cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	def load(self):
		filename = os.path.realpath('.') + "/dp/save/net.save"
		f = file(filename, 'rb')
		for p in params:
			p.set_value(cPickle.load(f))
		f.close()
