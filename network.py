import scipy.io as spio
import numpy as np
from numpy import array
import tensorflow as tf
import random

time_steps = 2200 #Input's length
input_dim = 1
label_len = 2
epoch = 50  # In every epoch we run all the inputs on the network once.
batch_size = 5  # Every batch_size we update the weights (with GPU recommand to use big batches)

def network(train_input, test_input, train_labels, test_labels):
    ## lstm network design
    data = tf.placeholder(tf.float32, [None, time_steps, input_dim]) #Maybe we want to change NONE to specific size
    target = tf.placeholder(tf.float32, [None, label_len])
    num_hidden = 80 #Number of hiiden layers: we can try different options
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
    val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1) #Shouts the error converting sparse..
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)
    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


    ##initialize array
    #init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)


    ##train network and test error at last epoch

    no_of_batches = int((len(train_input)) / batch_size)

    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+batch_size], train_labels[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(minimize,{data: inp, target: out})
        print ("Epoch ",str(i))
    #Test
    incorrect = sess.run(error,{data: test_input, target: test_labels})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    get_mistakes= (sess.run(mistakes, feed_dict={data: test_input, target: test_labels}))
    len_test = len(test_labels)
    mistake_list = []
    for i in range(0,len_test):
        if get_mistakes[i] == True:
            mistake_list.append(i)
    print(mistake_list)



