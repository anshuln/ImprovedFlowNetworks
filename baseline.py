import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


"""
	Potential Speedups: 

		32 float -> 16 float (64 complex to 32 complex )
		tf.fft3d -> tf.rfft3d 

"""


# get data 
(X, _), (X_test, _) = tf.keras.datasets.cifar10.load_data()
print(X.dtype)
shape = X[0].shape

# [ - 0.5 , 0.5  ]
X = X / 255 - 0.5 


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


	def __init__(self): 
		super(Conv, self).__init__()
		

	def call(self, X): 
		return X
		"""X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X / self.scale) 
		X = X * self.w
		X = tf.signal.ifft3d(X * self.scale ) 
		X = tf.math.real(X)
		return X"""

	def call_inv(self, X): 
		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X * self.scale ) # self.scale correctly 
		X = X / self.w
		X = tf.signal.ifft3d(X / self.scale)   
		X = tf.math.real(X)
		return X

	def log_det(self): 	return tf.math.reduce_sum(tf.math.log(tf.math.abs(self.w)))


	def build(self, input_shape): 
		print(input_shape)
		self.scale = np.sqrt(np.prod(input_shape[1:])) # np.sqrt(np.prod([a.value for a in input_shape[1:]]))

		# todo; change to [[[1, 0000],[0000], [000]] 

		def identitiy_initializer_real(shape, dtype=None): return tf.math.real(tf.signal.ifft3d(tf.ones(shape, dtype=tf.complex64)*self.scale)) 
		#def init(shape, dtype=None):
		#	zeros = np.zeros(shape) # TODO: explain why
		#	zeros[0,0,0] = 1 
		#	return zeros

		self.w 	= self.add_weight(shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True)
		print(self.w)
		self.w 	= tf.cast(self.w, dtype=tf.complex64)
		self.w 	= tf.signal.fft3d(self.w / self.scale)
		

	def compute_shape(self, input_shape): 
		print(input_shape)
		return input_shape[1:]


def nll(y_pred, y_true): 
	logdet = model.log_det()
	print(y_pred, y_true, logdet)

	normal = 0#normal_distributon(y_pred) 

	return logdet + normal
	

#input = tf.keras.Input(shape=shape)
#flow = Conv()(input)

#model = Model(inputs=input, outputs=input)

model = Sequential()
model.add(Conv())

model.compile(optimizer=tf.optimizers.Adam(0.001), loss=nll)

pred 	= model.predict(X[:2])
rec 	= np.add(model.predict_inv(pred), 0)

tensor = tf.ones([3,3])

print("-"*100)
print(tensor)
print(rec)

assert False

#print(rec[0,0,0,0])

model.summary()
fig, ax = plt.subplots(1, 3)

ax[0].imshow(X[0].reshape(32,32,3))
ax[1].imshow(pred[0].reshape(32,32,3))
ax[2].imshow(rec[0].reshape(32,32,3))





plt.pause(10**6)
		


