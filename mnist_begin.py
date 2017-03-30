import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input dimensions are 784x1
x = tf.placeholder(tf.float32, [None, 784])

# weights map input (784x1) to output (10x1)
W = tf.Variable(tf.zeros([784, 10]))
# bias for each output
b = tf.Variable(tf.zeros([10]))

# model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# correct labels
y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# @todo replace?

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

