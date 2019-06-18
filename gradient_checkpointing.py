import os

import tensorflow as tf 
import numpy as np 

from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
'''
TODO - Make a function to compute gradients effficiently in model class  -- TO BE Tested
	   This should call comput grads in all layers, make fucntions accordingly -- TO BE Tested
	   In main training loop, call apply_grads on model (using some optimizer)
	   Cleaning

Adapted version of this code - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/revnet
'''

std_dev = 2**8
batch_size = 64
def prettify(x): #Scales images between 0,1
	x = x - x.min()
	x = x / x.max()
	return x

class FlowSequential(tf.keras.Sequential):
	
	def __init__(self):
		super(FlowSequential,self).__init__()
	# 	self.saved_hidden = []   #Will store the input and activations of the last layer similar to gradient checkpointing 
	# 							 #TODO convert this to a tensor instead of list for GPU storage

	def loss_function(self,X):	#nll loss
		def log_normal_density(x): return tf.math.reduce_sum( -1/2 * (x**2/std_dev**2 + tf.math.log(2*np.pi*std_dev**2)) )

		log_det = self.log_det()
		normal = log_normal_density(X)
		return -(log_det+normal)

	def predict_inv(self,X):
		for layer in self.layers[::-1]:
			X = layer.call_inv(X)
		return X

	def get_last_inp(self,X,training=True):		#TODO - cleanup this function
		y1 = X
		for layer in self.layers[:-1]:
			y1 = layer.call(y1)

		return y1

	def log_det(self): 
		det = 0.
		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): continue 
			det = tf.add(det,layer.log_det())
		return det

	def compute_gradients(self,X):
		'''
		Computes gradients efficiently
		Returns - Tuple with first entry being list of grads and second loss
		'''
		x = self.get_last_inp(X)		#I think this records all operations onto the tape, thereby destroying purpose of checkpointing...
		last_layer = self.layers[-1]
		with tf.GradientTape() as tape:
			tape.watch(x)
			loss = self.loss_function(last_layer.call(x))	#May have to change
		grads_combined = tape.gradient(loss,[x]+last_layer.trainable_variables)
		dy, final_grads = grads_combined[0], grads_combined[1:]

		y=x
		intermediate_grads = []

		for layer in self.layers[-2:0:-1]:		#Not iterating over the first layer since we don't want that gradient to change according to input
			x = layer.call_inv(y)
			dy,grads = layer.backward_grads(x,dy)
			intermediate_grads = grads+intermediate_grads
			y = x 

		x = self.layers[0].call_inv(y)
		with tf.GradientTape() as tape:
			tape.watch(x)
			y = self.layers[0].call(x)
		init_grads = tape.gradient(y,self.layers[0].trainable_variables,output_gradients=dy)

		grads_all = init_grads + final_grads + intermediate_grads 	#Check ordering once, compare with model.trainable_variables
		print(len(init_grads),len(final_grads),len(intermediate_grads))
		return grads_all,loss

	def other_grads(self,X):
		with tf.GradientTape() as tape:
			loss = self.loss_function(self.call(X))
		grads = tape.gradient(loss,self.trainable_variables)
		return grads,loss

class LayerWithGrads(tf.keras.layers.Layer):	#Virtual Class
	def __init__(self):
		super(LayerWithGrads,self).__init__()

	def call(self,X):
		raise NotImplementedError

	def call_inv(self,X):
		raise NotImplementedError

	def backward_grads(self,x,dy):	
		with tf.GradientTape() as tape:
			tape.watch(x)
			y_ = self.call(x)	#Required to register the operation onto the gradient tape
		grads_combined = tape.gradient(y_,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]
		return dy,grads

class Conv(LayerWithGrads): 

	def __init__(self,trainable=True): 
		self.built = False
		super(Conv, self).__init__()
	

	def call(self, X): 
		if self.built == False:    #For some reason the layer is not being built without this line
			self.build(X.get_shape().as_list())

		#The next 2 lines are a redundant computation necessary because w needs to be an EagerTensor for the output to be eagerly executed, and that was not the case earlier
		#EagerTensor is required for backprop to work...
		#TODO - figure out a way to avoid, or open an issue with tf...
		self.w 	= tf.cast(self.w_real, dtype=tf.complex64)
		self.w 	= tf.signal.fft3d(self.w / self.scale)

		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X / self.scale) 
		X = X * self.w
		X = tf.signal.ifft3d(X * self.scale ) 
		X = tf.math.real(X)
		return X


	def call_inv(self, X): 
		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X * self.scale ) # self.scale correctly 
		#The next 2 lines are a redundant computation necessary because w needs to be an EagerTensor for the output to be eagerly executed, and that was not the case earlier
		self.w 	= tf.cast(self.w_real, dtype=tf.complex64)
		self.w 	= tf.signal.fft3d(self.w / self.scale)

		X = X / self.w

		X = tf.signal.ifft3d(X / self.scale)   
		X = tf.math.real(X)
		return X

	def log_det(self): 	return tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.signal.fft3d(tf.cast(self.w_real/self.scale,dtype=tf.complex64)))))	#Need to return EagerTensor


	def build(self, input_shape): 
		self.scale = np.sqrt(np.prod(input_shape[1:])) # np.sqrt(np.prod([a.value for a in input_shape[1:]]))

		# todo; change to [[[1, 0000],[0000], [000]] 

		def identitiy_initializer_real(shape, dtype=None):
			return (tf.math.real(tf.signal.ifft3d(tf.ones(shape, dtype=tf.complex64)*self.scale))) 

		self.w_real 	= self.add_variable(name="w_real",shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True)
		# self.w 	= tf.cast(self.w_real, dtype=tf.complex64)	#hacky way to initialize real w and actual w, since tf does weird stuff if 'variable' is modified
		# self.w 	= tf.signal.fft3d(self.w / self.scale)
		self.built = True

		

	def compute_output_shape(self, input_shape): 
		return tf.TensorShape(input_shape[1:])

def ReLU(x): return tf.math.maximum( x, 0 )


class UpperCoupledReLU(LayerWithGrads): 		
	def __init__(self,trainable=True): 
		self.built = False
		super(UpperCoupledReLU, self).__init__()

	def call(self, inputs): 		
		_, h, w, c = inputs.shape

		# assumes c is even
		assert c % 2 == 0, "The non-linearity assumes that c is even, it is not: c=%i"%c

		x1 = inputs[:, :, :, :c//2]
		x2 = inputs[:, :, :, c//2:]

		x2 = x2 + ReLU(x1)
		
		return tf.concat((x1, x2), axis=-1)


	def call_inv(self, outputs): 	
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



class Squeeze(LayerWithGrads): 
	def __init__(self,trainable=True): 
		self.built = False
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


def train_for_one_iter(model,X,optimizer,batch_size=batch_size):	#TODO implement mini-batch gradient descent here
	'''
	Trains the model on data for one iteration
	Args:
		model - model of class FlowSequential
		X - Data of type np.ndarray, having (num_samples,(sample_size)) dims
		optimizer - tf optimizer to use
		batch_size - int
	'''
	# num_batches = X.shape[0] // batch_size
	# X = np.random.permutation(X)
	# losses = []
	# for i in tqdm(range(0,X.shape[0]-X.shape[0]%batch_size,batch_size)):	
	# 	grads,loss = model.compute_gradients(X[i:(i+batch_size)])
	# 	optimizer.apply_gradients(zip(grads,model.trainable_variables))
	# 	losses.append(loss.numpy())
	# loss = np.mean(losses)	
	#Above code was for mini-batch gradient descent, but it gave weird results

	grads,loss = model.other_grads(X)
	g1, l1 = model.compute_gradients(X)
	print((np.array(g1))-np.array(grads))
	# print(np.array(grads))
	optimizer.apply_gradients(zip(grads,model.trainable_variables))
	loss = loss.numpy()
	
	return loss


def main():
	(X, _), (X_test, _) = tf.keras.datasets.cifar10.load_data() 	#This is actually numpy
	print(X.dtype)
	shape = X[0].shape


	# [ - 0.5 , 0.5  ]
	X = (X - 127.5)
	std_dev = 2**8

	model = FlowSequential()

	# model.add(UpperCoupledReLU())
	model.add(Conv())
	model.add(Squeeze())

	# model.add(LowerCoupledReLU())

	# model.add(Conv())
	# model.add(UpperCoupledReLU())

	# model.add(Conv())
	# model.add(LowerCoupledReLU())

	# model.add(Conv())
	# model.add(UpperCoupledReLU())

	# model.add(Conv())
	# model.add(LowerCoupledReLU())

	# model.add(Conv())
	# model.add(UpperCoupledReLU())

	# model.add(Conv())
	# model.add(LowerCoupledReLU())

	# model.add(Conv())
	# model.add(UpperCoupledReLU())

	# model.add(Conv())
	# model.add(LowerCoupledReLU())

	# print(model.layers)
	def loss_function(y1,y2):	#nll loss
		def log_normal_density(x): return tf.math.reduce_sum( -1/2 * (x**2/std_dev**2 + tf.math.log(2*np.pi*std_dev**2)) )

		log_det = model.log_det()
		normal = log_normal_density(y2)
		return -(log_det + normal)


	epochs = 100
	fast = 500
	optimizer = tf.optimizers.Adam(0.001)
	pred1 	= model.predict(X[:1])	#Don't remove,essential to build the layers...
	# model.compile(optimizer=tf.optimizers.Adam(0.001), loss=loss_function)
	# model.fit(X[:fast],X[:fast],epochs=epochs)
	losses = []
	for it in range(epochs):
		loss = train_for_one_iter(model,X[:fast],optimizer)
		print('Epoch : ',it,'Loss : ',loss)
		losses+=([loss])
	fixed_noise = tf.random.normal((1,16,16,12),0,std_dev)

	pred1 	= model.call(X[:2])
	rec1 = model.predict_inv(pred1)
	rec = model.predict_inv(fixed_noise[:1])
	plt.plot(np.arange(len(losses)),losses)
	plt.show()
	fig, ax = plt.subplots(1, 3)
	ax[0].imshow(prettify(X[1].reshape(32,32,3)))
	ax[1].imshow(prettify(rec1[1].numpy().reshape(32,32,3)))
	ax[2].imshow(prettify(rec.numpy()[0].reshape(32,32,3)))
	plt.show()

if __name__ == '__main__':
	main()

