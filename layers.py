from gradient_checkpointing import LayerWithGrads
import tensorflow as tf
import numpy as np
class Conv(LayerWithGrads): 

	def __init__(self,trainable=True): 
		self.built = False
		super(Conv, self).__init__()
	

	def call(self, X): 
		if self.built == False:    #For some reason the layer is not being built without this line
			self.build(X.get_shape().as_list())

		#The next 2 lines are a redundant computation necessary because w needs to be an EagerTensor for the output to be eagerly executed, and that was not the case earlier
		#EagerTensor is required for backprop to work...
		#Further, updating w_real will automatically trigger an update on self.w, so it is better to not store w at all
		#TODO - figure out a way to avoid, or open an issue with tf...
		self.w  = tf.cast(self.w_real, dtype=tf.complex64)
		self.w  = tf.signal.fft3d(self.w / self.scale)

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
		self.w  = tf.cast(self.w_real, dtype=tf.complex64)
		self.w  = tf.signal.fft3d(self.w / self.scale)

		X = X / self.w

		X = tf.signal.ifft3d(X / self.scale)   
		X = tf.math.real(X)
		return X

	def log_det(self):  return tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.signal.fft3d(tf.cast(self.w_real/self.scale,dtype=tf.complex64)))))    #Need to return EagerTensor


	def build(self, input_shape): 
		self.scale = np.sqrt(np.prod(input_shape[1:])) # np.sqrt(np.prod([a.value for a in input_shape[1:]]))

		# todo; change to [[[1, 0000],[0000], [000]] 

		def identitiy_initializer_real(shape, dtype=None):
			return (tf.math.real(tf.signal.ifft3d(tf.ones(shape, dtype=tf.complex64)*self.scale))) 

		self.w_real     = self.add_variable(name="w_real",shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True)
		# self.w    = tf.cast(self.w_real, dtype=tf.complex64)  #hacky way to initialize real w and actual w, since tf does weird stuff if 'variable' is modified
		# self.w    = tf.signal.fft3d(self.w / self.scale)
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

	def log_det(self):              return tf.zeros((1,)) 


class LowerCoupledReLU(UpperCoupledReLU): 
	def __init__(self,trainable=True): 
		self.built = False
		super(LowerCoupledReLU, self).__init__()

	def call(self, inputs):     return super(LowerCoupledReLU, self).call_inv(inputs)
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

	def log_det(self): return tf.zeros((1,))
