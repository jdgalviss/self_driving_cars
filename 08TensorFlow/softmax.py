# Solution is available in the other "solution.py" tab


#==================Classic Python============
import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    return np.exp(x)/np.sum(np.exp(x), axis = 0)

logits = [3.0, 1.0, 0.2]
print(softmax(logits))

#===================Tensorflow=============
import tensorflow as tf
def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        pass
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output