import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from opt_DFT import EfficientConv 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
	Potential Speedups: 

		32 float -> 16 float (64 complex to 32 complex )
		tf.fft3d -> tf.rfft3d - DONE, in a convoluted way

	Goals - See todo.txt

"""

# get data 
(X, _), (X_test, _) = tf.keras.datasets.cifar10.load_data()
print(X.dtype)
shape = X[0].shape


# [ - 0.5 , 0.5  ]
X = X / 1 - 127.5 
std_dev = 2**8

def prettify(x): #Scales images between 0,1
	x = x - x.min()
	x = x / x.max()
	return x

class Sequential(tf.keras.Sequential): 

	def predict_inv(self, X): 
		for layer in self.layers[::-1]: 
			X = layer.call_inv(X) 
		return X

	def log_det(self): 
		det = 0
		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): continue 
			det += layer.log_det()
		return det

class Conv(tf.keras.layers.Layer): 


	def __init__(self,trainable=True): 
		super(Conv, self).__init__()
		
	def call(self, X): 
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
		

	def compute_output_shape(self, input_shape): 
		return tf.TensorShape(input_shape[1:])

def ReLU(x): return tf.math.maximum( x, 0 )


class UpperCoupledReLU(tf.keras.layers.Layer): 


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

	def call(self, inputs): 	return super(LowerCoupledReLU, self).call_inv(inputs)
	def call_inv(self, outputs): return super(LowerCoupledReLU, self).call(outputs)			


class Squeeze(tf.keras.layers.Layer): 
	def build(self, input_shape): 
		_, self.h, self.w, self.c = input_shape

	def call(self, inputs):
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


def log_normal_density(x): return tf.math.reduce_sum( -1/2 * (x**2/std_dev**2 + tf.math.log(2*np.pi*std_dev**2)) )

def nll(y_true,y_pred): 	#TODO add scaling penalties?
	logdet = model1.log_det()
	print(y_pred, y_true, logdet)

	normal = log_normal_density(y_pred) 

	return -(logdet + normal)
	


model1 = Sequential()

model1.add(Squeeze())
model1.add(EfficientConv())
model1.add(UpperCoupledReLU())
model1.compile(optimizer=tf.optimizers.Adam(0.001), loss=nll)


pred 	= model1.predict(X[:2])

model1.summary()
model1.fit(X[:20],X[:20],epochs=1)

pred1 	= model1.predict(X[:2])
rec = model1.predict_inv(pred1)



fig, ax = plt.subplots(1, 3)

ax[0].imshow(prettify(X[1].reshape(32,32,3)))
ax[1].imshow(prettify(pred1[1].reshape(32,32,3)))
ax[2].imshow(prettify(rec.numpy()[1].reshape(32,32,3)))





plt.show()
		


