import os

import tensorflow as tf 
import numpy as np 

from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
std_dev = 2**8
'''
TODO - Make a function to compute gradients effficiently in model class  -- DONE
	   This should call comput grads in all layers, make fucntions accordingly -- DONE
	   Cleaning -- In Progress
	   intermediate layer grad computation -- DONE
	   iteratively update weights --DONE
Adapted version of this code - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/revnet
'''

class FlowSequential(tf.keras.Sequential):
	
	def __init__(self):
		super(FlowSequential,self).__init__()
	#   self.saved_hidden = []   #Will store the input and activations of the last layer similar to gradient checkpointing 
	#                            #TODO convert this to a tensor instead of list for GPU storage

	def loss_function(self,X):  #nll loss
		def log_normal_density(x): return tf.math.reduce_sum( -1/2 * (x**2/std_dev**2 + tf.math.log(2*np.pi*std_dev**2)) )

		log_det = self.log_det()    #This can be thought of as a regularizer...
		normal = log_normal_density(X)
		return -(log_det+normal) 

	def predict_inv(self,X):
		for layer in self.layers[::-1]:
			X = layer.call_inv(X)
		return X

	def get_last_inp(self,X,training=True):     #TODO - cleanup this function
		y1 = X
		for layer in self.layers[:-1]:
			# print('forward',y1[0])
			y1 = layer.call(y1)

		return y1

	def log_det(self): 
		det = 0.
		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): continue 
			det = tf.add(det,layer.log_det())
		return det

	def compute_and_apply_gradients(self,X,optimizer=None):
		'''
		Computes gradients efficiently and updates weights
		Returns - Loss on the batch
		'''
		x = self.call(X)        #I think putting this in context records all operations onto the tape, thereby destroying purpose of checkpointing...
		last_layer = self.layers[-1]
		#Computing gradients of loss function wrt the last acticvation
		with tf.GradientTape() as tape:
			tape.watch(x)
			loss = self.loss_function(x)    #May have to change
		grads_combined = tape.gradient(loss,[x])
		dy = grads_combined[0]
		y = x
		#Computing gradients for each layer
		for layer in self.layers[::-1]:     
			x = layer.call_inv(y)
			dy,grads = layer.compute_gradients(x,dy,layer.log_det)
			optimizer.apply_gradients(zip(grads,layer.trainable_variables))
			y = x 
		return loss

	def other_grads(self,X):
		'''
		Helper function to get 'True Gradients' of the network
		'''
		with tf.GradientTape() as tape:
			loss = self.loss_function(self.call(X))
		grads = tape.gradient(loss,self.trainable_variables)
		return grads,loss
	def train_for_one_epoch_data(self,X,optimizer,batch_size=32):
		'''
		Trains the model on data for one iteration
		Args:
			model - model of class FlowSequential
			X - Data of type np.ndarray, having (num_samples,(sample_size)) dims
			optimizer - tf optimizer to use
			batch_size - int
		'''
		num_batches = X.shape[0] // batch_size
		X = np.random.permutation(X)
		#Minibatch gradient descent
		for i in tqdm(range(0,X.shape[0]-X.shape[0]%batch_size,batch_size)):    
			# grads,loss = model.compute_gradients(X[i:(i+batch_size)])
			losses = []
			loss = self.compute_and_apply_gradients(X[i:(i+batch_size)],optimizer)
			losses.append(loss.numpy())
		loss = np.mean(losses)  

		# loss = model.compute_and_apply_gradients(X,optimizer)
		# loss = loss.numpy()
		
		return loss

	def train_for_one_epoch_generator(self,X,optimizer,num_batches):
		'''
		Trains model for an epoch using a tf.data.Dataset iter
		X- tf.data.Dataset.__iter__()
		'''
		for i in tqdm(range(num_batches)):
			losses = []
			loss = self.compute_and_apply_gradients(next(X),optimizer)
			losses.append(loss.numpy())
		loss = np.mean(losses)  

		return loss


class LayerWithGrads(tf.keras.layers.Layer):    #Virtual Class
	def __init__(self):
		super(LayerWithGrads,self).__init__()

	def call(self,X):
		raise NotImplementedError

	def call_inv(self,X):
		raise NotImplementedError

	def compute_gradients(self,x,dy,regularizer=None):  
		'''
		Computes gradients for backward pass
		Args:
			x - tensor compatible with forward pass, input to the layer
			dy - incoming gradient from backprop
			regularizer - function, indicates dependence of loss on weights of layer
		Returns
			dy - gradients wrt input, to be backpropagated
			grads - gradients wrt weights
		'''
		with tf.GradientTape() as tape:
			tape.watch(x)
			y_ = self.call(x)   #Required to register the operation onto the gradient tape
		grads_combined = tape.gradient(y_,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		if regularizer is not None:
			with tf.GradientTape() as tape:
				reg = -regularizer()
			grads_wrt_reg = tape.gradient(reg, self.trainable_variables)
			grads = [a[0]+a[1] for a in zip(grads,grads_wrt_reg)]
		return dy,grads

