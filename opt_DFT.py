import tensorflow as tf
import numpy as np

class EfficientConv(tf.keras.layers.Layer): 

	#Uses tf.rfft, which gives an output of h*w*c/2 instead of h*w*c. This means self.w is smaller by a factor of 2
	def __init__(self,trainable=True): 
		super(EfficientConv, self).__init__()
		
	def call(self, X): 
		X = tf.signal.rfft3d(X / self.scale) 
		X = X * self.w
		X = tf.signal.irfft3d(X * self.scale ) 
		X = tf.math.real(X)
		return X


	def call_inv(self, X): 
		# return X
		output_shape = X.shape
		X = tf.signal.rfft3d(X * self.scale ) # self.scale correctly 
		#The next line is a redundant computation necessary because w needs to be an EagerTensor for the output to be eagerly executed, and that was not the case earlier
		self.w 	= tf.signal.rfft3d(self.w / self.scale)

		X = X / self.w

		X = tf.signal.irfft3d(X / self.scale,output_shape)   
		X = tf.math.real(X)
		return X

	def log_det(self): 	return tf.math.reduce_sum(tf.math.log(tf.math.abs(self.w)))


	def build(self, input_shape): 
		self.scale = np.sqrt(np.prod(input_shape[1:])) # np.sqrt(np.prod([a.value for a in input_shape[1:]]))

		# todo; change to [[[1, 0000],[0000], [000]] 

		def identitiy_initializer_real(shape, dtype=None):
			shape_ones = [shape[0],shape[1],(shape[2]//2 + 1)]	#irfft needs input to be of length fft_length/2 + 1
			return (tf.math.real(tf.signal.irfft3d(tf.ones(shape_ones, dtype=tf.complex64)*self.scale,shape))) 

		self.w_real 	= self.add_variable(name="w_real",shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True)
		#hacky way to initialize real w and actual w, since tf does weird stuff if 'variable' is modified
		self.w 	= tf.signal.rfft3d(self.w_real / self.scale)
		

	def compute_output_shape(self, input_shape): 
		return tf.TensorShape(input_shape[1:])
