import import_data as import_data
import network
import tensorflow as tf

shuffle_flag = 1

if __name__ == "__main__":
    #Call from import_data module the function: load_data_mat.
    train_input, test_input, train_labels, test_labels = import_data.load_data_mat(import_data.path, shuffle_flag, 1)
    #Save the data that we used (with the specific order)
    import_data.save_data_py(train_input, test_input, train_labels, test_labels)
    #Test to see if we saved successfuly
    train_input, test_input, train_labels, test_labels =  import_data.load_data_py()
    #Run the network itself
    network.network(train_input, test_input ,  train_labels, test_labels)



