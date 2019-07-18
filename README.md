# ImprovedFlowNetworks
Some tricks to improve flow networks

The following are implemented - 
1. Faster fft of real valued vectors by exploiting complex symmetry of the fourier transforms. 
	* The major contribution here was to register gradients for `tf.irfft3d` and writing an implementation for `rfft3d`
2. Gradient checkpointing for invertible networks allowing for constant memory backprop
	* `gradient_checkpointing.py` contains relevant layer and model classes which can be generalized to other models.
3. Variational dequantization according to [this](https://arxiv.org/abs/1902.00275) paper.
	* `dequantization.py` contains the `FlowWithDequant` class which can wrap around any flow to make it dequantized.
4. Faster data loaders for images using `tf.data.Dataset`.
