import tensorflow as tf
import numpy as np

# TODO - put this here instead of in tf, and verify that it is correct - 
#			tf.RegisterGradient('IRFFT3D')()
#TODO - understand why w being rfft'd gives error but not X...
#TODO - Implement RFFT ?
#TODO - Benchmark
class EfficientConv(tf.keras.layers.Layer): 

	#Uses tf.rfft, which gives an output of h*w*c/2 instead of h*w*c. This means self.w is smaller by a factor of 2
	def __init__(self,trainable=True): 
		super(EfficientConv, self).__init__()
	
	def rfft3d(self,X,inp_shape):
		'''
		Args : X - Tensor of 3 or 4 dims
			   inp_shape - shape of X as a list
		'''
		fft_length = inp_shape[-1]//2 + 1

		if len(inp_shape) == 3:		#Quick and dirty solution to problem of first shape val being none, TODO make it generalized
			return (tf.signal.fft3d(tf.cast(X,dtype=tf.complex64)))[:,:,:fft_length]

		elif len(inp_shape) == 4:
			return (tf.signal.fft3d(tf.cast(X,dtype=tf.complex64)))[:,:,:,:fft_length]

		else:
			raise NotImplementedError



	# def irfft3d(self,X,shape):


	def call(self, X): 
		output_shape = X.shape
		X = tf.signal.rfft3d(X / self.scale) 
		X = X * self.w
		X = tf.signal.irfft3d(X * self.scale,output_shape[1:] ) 
		X = tf.math.real(X)
		print(self.w.shape,X.shape)
		return X


	def call_inv(self, X): 
		# return X
		output_shape = X.shape
		X = tf.signal.rfft3d(X * self.scale ) # self.scale correctly
		#The next line is a redundant computation necessary because w needs to be an EagerTensor for the output to be eagerly executed, and that was not the case earlier
		self.w 	= self.rfft3d(self.w_real / self.scale,self.w_real.get_shape().as_list())

		X = X / self.w
		print('InverseComputation',output_shape,X.shape )
		X = tf.signal.irfft3d(X / self.scale,output_shape[1:])   
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
		self.w 	= self.rfft3d(self.w_real / self.scale,self.w_real.get_shape().as_list())
		

	def compute_output_shape(self, input_shape): 
		return tf.TensorShape(input_shape[1:])
