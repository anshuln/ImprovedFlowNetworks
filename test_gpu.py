import tensorflow as tf
import time 

a = tf.random.normal((10000, 10000), 0, 1)


t0 = time.time()
with tf.device("cpu"): a @ a 
cpu = time.time() - t0
print("CPU: \t", cpu)


t0 = time.time()
a @ a 
t1 = time.time()

gpu = t1- t0
print("GPU: \t", gpu)


print("Speedup: \t", cpu / gpu)
