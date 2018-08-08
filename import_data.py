import scipy.io as spio
import numpy as np
from numpy import array
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import pickle


global path
path= r'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\mat_to_python_st.mat'

TRAIN_INPUT = 'train_input_st'
TRAIN_LABELS = 'train_labels_st'
TEST_LABELS = 'test_labels_st'
TEST_INPUT = 'test_input_st'

time_steps = 2200

#Function which suffle labels and inputs.
def unison_shuffled_copies(a,b):
    '''
     array a and b  can have different shapes, but with the same length (leading dimension).
    :param a: array 1
    :param b: array 2
    :return: shuffle each of them, such that corresponding elements continue to correspond
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# type_input: 0 = ECG , 1 = GSR , 2 = both
def load_data_mat(path, shuffle_flag,type_input=2):
    """
    :param path: path of mat file with variables
    :param shuffle_flag: if shuffle_flag ==1 than shuffle all variables else no shuffle
    :return:
    """
    mat = spio.loadmat(path, squeeze_me=True)
    train_input =mat[TRAIN_INPUT]  # array
    train_labels = mat[TRAIN_LABELS] # structure containing an array
    test_labels = mat[TEST_LABELS]# array of structures
    test_input = mat[TEST_INPUT]

    ## shuffle train and test
    if (shuffle_flag ==1):
        train_input,train_labels= unison_shuffled_copies(train_input,train_labels)
        test_input,test_labels= unison_shuffled_copies(test_input,test_labels)
    # input becomes a list of numpy 2d arrays, labels becomes a list of lists

    if (type_input!=2):
        train_input = [np.reshape(a[type_input,:],(time_steps,1)) for a in train_input]
        test_input = [np.reshape(a[type_input,:],(time_steps,1)) for a in test_input]
        train_labels = [a.tolist() for a in train_labels]
        test_labels = [a.tolist() for a in test_labels]
    else:
        train_input = [np.transpose(a) for a in train_input]
        test_input = [np.transpose(a) for a in test_input]
        train_labels = [a.tolist() for a in train_labels]
        test_labels = [a.tolist() for a in test_labels]

    return  train_input, test_input, train_labels, test_labels

##### save and load variables
#Load .npy files (for saving the vars we already loaded)
def load_data_py():
    """
    get all variables saved in npy files
    :return:
    """
    try:
        train_input= np.load("train_input.npy")
        test_input= np.load("test_input.npy")
        train_labels= np.load("train_labels.npy")
        test_labels= np.load("test_labels.npy")
    except:
        train_input, test_input, train_labels, test_labels = [], [], [], []
    return train_input, test_input ,  train_labels, test_labels

#Save to .npy files. We'll load those later
def save_data_py(train_input,test_input, train_labels, test_labels ):
    """
    save  variables in function arguments to npy files

    """
    np.save('train_input.npy', train_input)
    np.save('test_input.npy', test_input)
    np.save('train_labels.npy', train_labels)
    np.save('test_labels.npy', test_labels)
