import tensorflow as tf 
import numpy as np 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
TODO - Make a function to compute gradients effficiently in model class  -- TO BE Tested
	   This should call comput grads in all layers, make fucntions accordingly -- TO BE Tested
	   In main training loop, call apply_grads on model (using some optimizer)

Adapted version of this code - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/revnet
'''

std_dev = 2**8
class FlowSequential(tf.keras.Sequential):
	
	# def __init__(self):
	# 	super(FlowSequential,self).__init__()
	# 	self.saved_hidden = []   #Will store the input and activations of the last layer similar to gradient checkpointing 
	# 							 #TODO convert this to a tensor instead of list for GPU storage

	def loss_function(self,X):	#nll loss
		def log_normal_density(x): return tf.math.reduce_sum( -1/2 * (x**2/std_dev**2 + tf.math.log(2*np.pi*std_dev**2)) )

		log_det = self.log_det()
		normal = log_normal_density(X)
		return -(logdet + normal)

	def predict_inv(self,X):
		for layer in self.layers[::-1]:
			X = layer.call_inv(X)
		return X

	def _call(self,X,training=True):		#Hack is to differentiate it from predict, and to build the model....
		self.saved_hidden = []
		self.saved_hidden.append(X)
		for layer in self.layers:
			X = layer.call(X)
		self.saved_hidden.append(X)

	def log_det(self): 
		det = 0
		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): continue 
			det += layer.log_det()
		return det

	def compute_gradients(self):
		'''
		Computes gradients effieciently
		Returns - Tuple with first entry being list of grads and second loss
		'''
		y_fin = self.saved_hidden[-1]
		last_layer = self.layers[-1]
		print(y_fin)
		x = last_layer.call_inv(y_fin)
		with tf.GradientTape() as tape:
			tape.watch(x)
			loss = self.loss_function(y_fin)	#May have to change
		grads_combined = tape.gradient(loss,[x]+last_layer.trainable_variables)
		dy, final_grads = grads_combined[0], grads_combined[1:]

		y=x

		intermediate_grads = []
		for layer in self.layers[-2:0:-1]:		#Not iterating over the first layer since we don't want that gradient to change according to input
			x = layer.call_inv(y)
			dy,grads = layer.backward_grads(x,y,dy)
			intermediate_grads = grads+intermediate_grads
			y = x 

		with tf.GradientTape() as tape:
			init_grads = tape.gradient(y,self.layers[0].trainable_variables,output_gradients=dy)

		grads_all = init_grads + final_grads + intermediate_grads 	#Check ordering once, compare with model.trainable_variables

		return grads_all,loss

class LayerWithGrads(tf.keras.layers.Layer):	#Virtual Class
	def __init__(self):
		super(LayerWithGrads,self).__init__()

	def call(self,X):
		raise NotImplementedError

	def call_inv(self,X):
		raise NotImplementedError

	def backward_grads(self,x,y,dy):
		with tf.GradientTape() as tape:
			tape.watch(x)
		grads_combined = tape.gradient(y,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		return dy,grads

class Conv(tf.keras.layers.Layer): 

	def backward_grads(self,x,y,dy):
		with tf.GradientTape() as tape:
			tape.watch(x)
		grads_combined = tape.gradient(y,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		return dy,grads

	def __init__(self,trainable=True): 
		self.built = False
		super(Conv, self).__init__()
	

	def call(self, X): 
		if self.built == False:    #For some reason the layer is not being built without this line
			self.build(X.get_shape().as_list())

		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X / self.scale) 
		X = X * self.w
		X = tf.signal.ifft3d(X * self.scale ) 
		X = tf.math.real(X)
		return X


	def call_inv(self, X): 
		# return X
		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X * self.scale ) # self.scale correctly 
		#The next 2 lines are a redundant computation necessary because w needs to be an EagerTensor for the output to be eagerly executed, and that was not the case earlier
		self.w 	= tf.cast(self.w_real, dtype=tf.complex64)
		self.w 	= tf.signal.fft3d(self.w / self.scale)

		X = X / self.w

		X = tf.signal.ifft3d(X / self.scale)   
		X = tf.math.real(X)
		return X

	def log_det(self): 	return tf.math.reduce_sum(tf.math.log(tf.math.abs(self.w)))


	def build(self, input_shape): 
		self.scale = np.sqrt(np.prod(input_shape[1:])) # np.sqrt(np.prod([a.value for a in input_shape[1:]]))

		# todo; change to [[[1, 0000],[0000], [000]] 

		def identitiy_initializer_real(shape, dtype=None):
			return (tf.math.real(tf.signal.ifft3d(tf.ones(shape, dtype=tf.complex64)*self.scale))) 
		#def init(shape, dtype=None):
		#	zeros = np.zeros(shape) # TODO: explain why
		#	zeros[0,0,0] = 1 
		#	return zeros

		self.w_real 	= self.add_variable(name="w_real",shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True)
		self.w 	= tf.cast(self.w_real, dtype=tf.complex64)	#hacky way to initialize real w and actual w, since tf does weird stuff if 'variable' is modified
		self.w 	= tf.signal.fft3d(self.w / self.scale)
		self.built = True

		

	def compute_output_shape(self, input_shape): 
		return tf.TensorShape(input_shape[1:])

def ReLU(x): return tf.math.maximum( x, 0 )


class UpperCoupledReLU(tf.keras.layers.Layer): 
	def __init__(self,trainable=True): 
		self.built = False
		super(UpperCoupledReLU, self).__init__()

	def backward_grads(self,x,y,dy):
		with tf.GradientTape() as tape:
			tape.watch(x)
		grads_combined = tape.gradient(y,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		return dy,grads

	def call(self, inputs): 		
		_, h, w, c = inputs.shape

		# assumes c is even
		assert c % 2 == 0, "The non-linearity assumes that c is even, it is not: c=%i"%c

		x1 = inputs[:, :, :, :c//2]
		x2 = inputs[:, :, :, c//2:]

		x2 = x2 + ReLU(x1)
		
		return tf.concat((x1, x2), axis=-1)


	def call_inv(self, outputs): 	
		print(outputs.get_shape())
		_, h, w, c = outputs.shape

		# assumes c is even
		assert c % 2 == 0, "The non-linearity assumes that c is even, it is not: c=%i"%c

		x1 = outputs[:, :, :, :c//2]
		x2 = outputs[:, :, :, c//2:]

		x2 = x2 - ReLU(x1)
		
		return tf.concat((x1, x2), axis=-1)

	def log_det(self): 				return 0. 


class LowerCoupledReLU(UpperCoupledReLU): 
	def __init__(self,trainable=True): 
		self.built = False
		super(LowerCoupledReLU, self).__init__()

	def call(self, inputs): 	return super(LowerCoupledReLU, self).call_inv(inputs)
	def call_inv(self, outputs): return super(LowerCoupledReLU, self).call(outputs)			

	def backward_grads(self,x,y,dy):
		with tf.GradientTape() as tape:
			tape.watch(x)
		grads_combined = tape.gradient(y,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		return dy,grads


class Squeeze(tf.keras.layers.Layer): 
	def __init__(self,trainable=True): 
		# self.built = False
		super(Squeeze, self).__init__()
	

	def build(self, input_shape): 
		_, self.h, self.w, self.c = input_shape
		self.built = True

	def call(self, inputs):
		if self.built == False:    #For some reason the layer is not being built without this line
			self.build(inputs.get_shape().as_list())
		h, w, c = self.h, self.w, self.c
		out = tf.reshape(  inputs,     (-1, h//2, 2, w//2, 2, c))
		out = tf.transpose(out,        (0, 1, 3, 2, 4, 5))
		return tf.reshape(out,      (-1, h//2, w//2, c * 2 * 2))

	def call_inv(self, inputs):
		h, w, c = self.h, self.w, self.c
		out = tf.reshape(    inputs, (-1, h//2, w//2, 2, 2, c))
		out = tf.transpose(  out,    (0, 1, 3, 2, 4, 5))
		out = tf.reshape(    out,    (-1, h, w, c))
		return out 

	def compute_output_shape(self, input_shape): 
		n, h, w, c = input_shape
		return (n, h//2, w//2, c*4)

	def log_det(self): return 0.

	def backward_grads(self,x,y,dy):
		with tf.GradientTape() as tape:
			tape.watch(x)
		grads_combined = tape.gradient(y,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		return dy,grads


def train_for_one_iter(model,X,optimizer):
	y = model._call(X)	#Updates saved_hidden
	grads,loss = model.compute_gradients()
	optimizer.apply_gradients(grads,model.trainable_variables)

	return y,loss


def main():
	(X, _), (X_test, _) = tf.keras.datasets.cifar10.load_data()
	print(X.dtype)
	shape = X[0].shape


	# [ - 0.5 , 0.5  ]
	X = X / 1 - 127.5 
	std_dev = 2**8

	model = FlowSequential()

	model.add(Squeeze())
	model.add(Conv())
	model.add(UpperCoupledReLU())

	print(model.layers)

	epochs = 10
	fast = 100
	optimizer = tf.optimizers.Adam(0.001)
	pred1 	= model.predict(X[:1])	#Don't remove,essential to build the layers...
	for it in range(epochs):
		train_for_one_iter(model,X[:fast],optimizer)

	fixed_noise = tf.random.normal((1,16,16,12),0,std_dev)

	pred1 	= model.call(X[:2])
	rec = model.predict_inv(fixed_noise[:1])

	fig, ax = plt.subplots(1, 3)
	print(rec.get_shape())
	ax[0].imshow(prettify(X[1].reshape(32,32,3)))
	ax[1].imshow(prettify(pred1[1].reshape(32,32,3)))
	ax[2].imshow(prettify(rec.numpy()[0].reshape(32,32,3)))
	plt.show()

main()

