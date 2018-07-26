import scipy.io as spio
import numpy as np
from numpy import array
import tensorflow as tf
import random


def unison_shuffled_copies(a,b):
    '''
     array a and b sifferent shapes, but with the same length (leading dimension).
    :param a: array 1
    :param b: array 2
    :return: shuffle each of them, such that corresponding elements continue to correspond
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

## load data from matlab
mat = spio.loadmat(r'C:\Users\User\Documents\limudim\project\train_test_forPython\mat_to_python.mat', squeeze_me=True)
train_input =mat['train_input']  # array
train_labels = mat['train_labels'] # structure containing an array
test_labels = mat['test_labels']# array of structures
test_input = mat['test_input']

## shuffle train and test
train_input,train_labels= unison_shuffled_copies(train_input,train_labels)
test_input,test_labels= unison_shuffled_copies(test_input,test_labels)

## input becomes a list of numpy 2d arrays, labels becomes a list of lists
train_input = [np.transpose(a) for a in train_input]
test_input = [np.transpose(a) for a in test_input]
train_labels = [a.tolist() for a in train_labels]
test_labels = [a.tolist() for a in test_labels]

print ("test and training data loaded")

## lstm network design
data = tf.placeholder(tf.float32, [None, 2200,2]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 6])
num_hidden = 80
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


##initialize array
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

##train network and test error at last epoch
batch_size = 10
no_of_batches = int((len(train_input)) / batch_size)
epoch = 200
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_labels[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print ("Epoch ",str(i))
incorrect = sess.run(error,{data: test_input, target: test_labels})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
