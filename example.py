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
		X,num_batches = load_images_dataset('Data')
		iter = X.make_one_shot_iterator()
		loss = model.train_for_one_epoch_generator(X,num_batches)
		print('Epoch {}, loss {}'.format(i,loss))




