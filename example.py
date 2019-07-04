from layers import *
from gradient_checkpointing import FlowSequential
from dequantization import FlowWithDequant
from load_data import *

if __name__ == "__main__":
	model = FlowSequential()
	model.add(Squeeze())
	model.add(Conv())
	model.add(UpperCoupledReLU())
	model.add(Conv())
	model.add(LowerCoupledReLU())

	for i in range(5):	#Epochs
		X,num_batches = load_image_dataset('Data')
		iterator = iter(X)
		optimizer = tf.optimizers.Adam()
		loss = model.train_for_one_epoch_generator(iterator,optimizer,num_batches)
		print('Epoch {}, loss {}'.format(i,loss))




