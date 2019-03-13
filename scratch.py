# here, simple perceptron learns to understand whether input X is a symmetric vector
# it works
# though in this very simple version, it works on really random vectors X
# it's certainly much harder to distinguish almost-symmetric vector X  (if X is random and it's dim is big, then almost-symmetry is a rare thing)

import tensorflow as tf
import numpy as np
import os

hvs = 5 #half input vector size
learning_rate = 0.1
epoch = 500000
#n_input = 2
n_hidden = 10
n_output = 1

X = tf.placeholder(tf.float32, shape=[None, 2*hvs])
Y = tf.placeholder(tf.float32)
L2 = tf.layers.dense(inputs = X, units = n_hidden, activation = tf.nn.relu, kernel_initializer = tf.initializers.glorot_normal(), bias_initializer = tf.random_normal_initializer(), name='L2')
#L3 = tf.layers.dense(inputs = L2, units = n_hidden, activation = tf.nn.sigmoid, kernel_initializer = tf.initializers.glorot_normal(), bias_initializer = tf.random_normal_initializer(), name='L3')
#L4 = tf.layers.dense(inputs = L3, units = n_hidden, activation = tf.nn.relu, kernel_initializer = tf.initializers.glorot_normal(), bias_initializer = tf.random_normal_initializer(), name='L4')
hy = tf.layers.dense(L2, 1, activation=tf.nn.sigmoid, bias_initializer = tf.random_normal_initializer(), name="hy")
#hy = tf.sigmoid(tf.matmul(L2, W2) + b2)
cost = tf.losses.mean_squared_error(Y, hy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=0.0001).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
#session.run(optimizer, feed_dict={X: x_data, Y: y_data})

def print_weights():
  vars = tf.trainable_variables()
  print(vars)  # some infos about variables...
  vars_vals = session.run(vars)
  for var, val in zip(vars, vars_vals):
    print("var: {}, value: {}".format(var.name, val))



with tf.Session() as session:
  session.run(init)
  for step in range(epoch):
    x_data = np.array([]).reshape(0,2*hvs)
    y_data = np.array([])
    for i in range(1):
      if np.random.rand() > 0.5:
        rnd_vec = np.random.randint(2, size=(1, hvs))
        add_vec = np.concatenate((rnd_vec, rnd_vec), axis=1)
        x_data = np.concatenate((x_data, add_vec), axis=0)
        y_data = np.concatenate((y_data, np.array([0])), axis=0)
        #y_data = np.concatenate((y_data, [x_data[i][0] + x_data[i][1]]), axis=0)
      else:
        add_vec = np.random.randint(2, size=(1, 2*hvs))
        while np.array_equal(add_vec[0][:hvs], add_vec[0][hvs:]):
            add_vec = np.random.randint(2, size=(1, 2 * hvs))
        x_data = np.concatenate((x_data, add_vec), axis=0)
        y_data = np.concatenate((y_data, np.array([1])), axis=0)
        #y_data = np.concatenate((y_data, [x_data[i][0] + x_data[i][1]]), axis=0)
    session.run(optimizer, feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
      #weights = tf.get_default_graph().get_tensor_by_name(os.path.split(L2.name)[0] + '/kernel:0')
      #tf.print(weights, [weights])
      print_weights()
      print("x_data = ", x_data)
      print("y_data = ", y_data)
      print(session.run(cost, feed_dict={X: x_data, Y: y_data}))
      learning_rate = learning_rate * 0.98
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
      #print("x_data = ", x_data)
      #print("y_data = ", y_data)
  answer = tf.equal(tf.floor(hy+0.5), Y)
  accuracy = tf.reduce_mean(tf.cast(answer, "float"))
  print(session.run([hy], feed_dict={X: x_data, Y: y_data}))
  print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))

