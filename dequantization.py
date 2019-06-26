import tensorflow as tf
import numpy as np

from gradient_checkpointing import FlowSequential, prettify
from layers import *

# TODO - get conditional flows to work, currently dequantization is
# independent of sample


class Dequantizer(FlowSequential):
	'''
	Flow model for dequantizng the input images
	'''

	def __init__(self, Layers):
		super(Dequantizer,self).__init__()
		for layer in Layers:
			self.add(layer)

	def call(self, X):
		#TODO cleanup, and figure out a way to deal with none shapes better
		if X.shape[0] == None:
			epsilon = tf.expand_dims(tf.random.normal(X.shape[1:], 0, 1),0)
		else:
			epsilon = tf.random.normal(X.shape, 0, 1) 
		for layer in self.layers:
			epsilon = layer.call(epsilon)
		if X.shape[0] == None:
			return X + tf.expand_dims(tf.reshape(epsilon,X.shape[1:]),0)
		else:
			return X + tf.reshape(epsilon,X.shape)

class FlowWithDequant(FlowSequential):

	def __init__(self, actualFlow,dequantFlowLayers=[Squeeze(), Conv(),UpperCoupledReLU(),LowerCoupledReLU()]):
		super(FlowWithDequant,self).__init__()
		self.dequantFlow = Dequantizer(dequantFlowLayers)
		self.actualFlow = actualFlow
		self.layers = self.dequantFlow.layers + self.actualFlow.layer

	def call(self, X):
		return self.actualFlow.call(self.dequantFlow.call(X))
		# Dequant flow returns X+U

	def predict_inv(self, X):
		return self.actualFlow.predict_inv(X)

	def loss_function(self, X):
		# Ignoring the p(epsilon) term in the denominator, since it is constant in expectation,
		# And does not depend on the model
		return self.actualFlow.loss_function(X) - self.dequantFlow.log_det()

	def compute_and_apply_gradients(self,X,optimizer):
		#Need to rewrite this function because self.layers evals to the two 
		#models inside and not actual layers.
		#TODO change self.layers so that we do not need this function
		x = self.call(
			X)  
		with tf.GradientTape() as tape:
			tape.watch(x)
			loss = self.loss_function(x)  # May have to change
		grads_combined = tape.gradient(loss, [x])
		dy = grads_combined[0]
		y = x
		# Computing gradients for each layer
		for layer in (self.dequantFlow.layers+self.actualFlow.layers)[::-1]:
			x = layer.call_inv(y)
			dy, grads = layer.compute_gradients(x, dy, layer.log_det)
			optimizer.apply_gradients(zip(grads, layer.trainable_variables))
			y = x
		return loss
