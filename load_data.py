import tensorflow as tf

def load_image_dataset(folder, new_size=(64, 64),batch_size=32):
	def _parse_function(filename):
		image_string = tf.io.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string)
		image_resized = tf.image.resize(image_decoded, new_size)
		return image_resized

	files = ['{}/{}'.format(folder,f) for f in os.listdir(
		folder) if os.path.isfile(os.path.join(folder, f))]
	dataset = tf.data.Dataset.from_tensor_slices(tf.constant(files))
	dataset = dataset.shuffle(buffer_size=100)
	dataset = dataset.map(map_func=_parse_function,num_parallel_calls=4)
	dataset = dataset.prefetch(buffer_size=32)
	dataset = dataset.batch(batch_size=batch_size)
	return dataset,(len(files)//batch_size)
