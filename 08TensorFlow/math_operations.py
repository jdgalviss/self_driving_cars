# Solution is available in the "solution.ipynb" 
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))
output = None
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(z)
    print(output)