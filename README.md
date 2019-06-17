# ImprovedFlowNetworks
Some tricks to improve flow networks

The following are implemented - 
1. Faster fft of real valued vectors by exploiting complex symmetry of the fourier transforms. 
	* The major contribution here was to register gradients for `tf.irfft3d` and writing an implementation for `rfft3d`
2. Gradient checkpointing for invertible networks allowing for constant memory backprop
	* `gradient_checkpointing.py` contains relevant layer and model classes which can be generalized to other models.