import os
from tsfresh import extract_features, extract_relevant_features
from tsfresh import select_features
import numpy as np
from tsfresh.utilities.dataframe_functions import impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from Classification import *
from tsfresh.feature_extraction.settings import from_columns
import random
import pickle

class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn including normalization to make it
    compatible with pandas DataFrames.
    """

    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = ["pca_{}".format(i) for i in range(len(pandas_df.columns))]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X is not compatible with the
                               columns from the previous X data
        """
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError("The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X

def removeLevel3(df_data, df_labels,clasification_type):
    levels = [1,2,4,5]
    levels_no=[3]
    df_labels = df_labels.loc[df_labels['level'].isin(levels)]
    array_good_id =df_labels['id'].values
    df_data = df_data.loc[df_data['id'].isin(array_good_id)]
    if clasification_type == "CL2Condition2":
        tmp_df = df_labels.loc[(df_labels['condition'] == 0)]
        tmp_df = tmp_df.replace({'levelB_condition': 2}, 1)
        tmp_df = tmp_df.replace({'levelB_condition': 3}, 0)
        tmp_df = tmp_df.replace({'levelB_condition': 4}, 2)
        tmp_df = tmp_df.replace({'levelB_condition': 5}, 2)
        df_labels.update(tmp_df)
        tmp_df = df_labels.loc[(df_labels['condition'] == 1)]
        tmp_df = tmp_df.replace({'levelB_condition': 3}, 0)
        tmp_df = tmp_df.replace({'levelB_condition': 1}, 3)
        tmp_df = tmp_df.replace({'levelB_condition': 2}, 3)
        tmp_df = tmp_df.replace({'levelB_condition': 5}, 4)
        df_labels.update(tmp_df)
    df_data.to_csv(r'C:\Users\User\Documents\limudim\project\python_code\full_flow\filtered_features_all_data.csv')
    df_labels.to_csv(r'C:\Users\User\Documents\limudim\project\python_code\full_flow\all_subjects_labels_updated.csv')
    return df_data, df_labels


#between_shuffle = 1 <=> shuffle with all subjects train and test. between_shuffle = 0 <=> subjects different in train and in test
# clasification_type = ["CL5Levels", "CL2levels", "CL2Condition2", "condition2"]
def get_train_test(df_data, df_label, ratio, between_shuffle=1, classification_type = 'CL5Levels'):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    y_col = 'level'
    if classification_type == 'condition2':
        y_col = 'condition'
    if classification_type == 'CL2levels':
        y_col = 'levelB'
    if classification_type == 'CL2Condition2':
        y_col = 'levelB_condition'
    if between_shuffle == 1:
        mask = np.random.choice([True, False], size=(len(df_label),), p=[ratio, 1 - ratio])
        X_train = df_data[mask].values
        X_test = df_data[~mask].values
        Y_train = df_label[mask]
        Y_test = df_label[~mask]
        Y_train = Y_train[y_col].values
        Y_test = Y_test[y_col].values
        return  X_train, Y_train, X_test, Y_test
    else:
    ##does not work!!!!
        subjects = df_label['sub'].unique()

        random.shuffle(subjects)
        num_subjects = len(subjects)
        num_train = int(round(num_subjects*ratio))
        train_sub = subjects[0:num_train]
        test_sub = subjects[num_train:num_subjects]

        df_labels_train = df_label.loc[df_label['sub'].isin(train_sub)]
        train_array_good_id = df_labels_train['id'].values
        index = df_labels_train.index[df_label['sub'].isin(train_sub)].tolist()

        df_data_train = df_data.loc[df_data['id'].isin(train_array_good_id)]

        df_labels_test = df_label.loc[df_label['sub'].isin(test_sub)]
        test_array_good_id = df_labels_test['id'].values
        df_data_test = df_data.loc[df_data['id'].isin(test_array_good_id)]
        # get x y
        Y_train = df_labels_train[y_col].values
        Y_test = df_labels_test[y_col].values
        X_train = df_data_train.values
        X_test = df_data_test.values
        return  X_train, Y_train, X_test, Y_test

# global to_filter_by

# if __name__ == '__main__':

    # parameter for training
    ratio = 0.7
    # clasification_type = ["CL5Levels", "CL2levels", "CL2Condition2", "condition2"]


classification_type = 'condition2'
shuffle = 1  # (1 = train and test on same subjects, 0 = test on new subjects, not supported)


# load  data in a table containing all inputs and all features calculated
extracted_features_original = pd.read_csv(r'C:\Users\User\Documents\2017-2018\Project\network\current_use\all_features_original.csv')

#load labels with the following coulmns:
    # sub num
    # sub id (a different number for each time window,
    #condition (stress/no stress)
    # level(1-5)
    # levelB( levels 1-3 are 0, level 4-5 is 1)
    #levelB_condition = level (in removeLevel3 function gets new values
impute(extracted_features_original)  # takes care of nan and similar (in place)

#All labels
labels = pd.read_csv(r'C:\Users\User\Documents\2017-2018\Project\network\current_use\all_subjects_labels_original.csv')


# adjust data and labels to remove level 3 rows (No using for now)
if (classification_type == ("CL2levels" or "CL2Condition2")):
    extracted_features_original, labels =  removeLevel3(extracted_features_original, labels, classification_type)

y_for_filtering = labels.levelB #will use to get relevant features

# rename data
global x_input
x_input = extracted_features_original
y_labels = y_for_filtering #?



def classify():
    #####################################################################################
    ###
    ###             classification
    ######################################################################################
    #We train the models:
    #Save them in files.
    #finialized ... for levels
    #st_finialized ... for conditions
    #Danielle
    ratio = 0.7
    no_classifiers = 7
    between_shuffle = 1



    #Train and learn for conditions prediction
    X_data = X_filtered_features
    y_labels = labels
    classification_type = 'condition2'

    #get train and test data and labels according to ratio and classification_type
    X_train, Y_train, X_test, Y_test =  get_train_test(X_data, y_labels, ratio, between_shuffle=between_shuffle, classification_type=classification_type)
    #train several models
    dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = no_classifiers, classification_type=classification_type)
    print('finish train for condition2')
    # Display results
    display_dict_models(dict_models)

    # Train and learn for Clevels prediction
    X_data = X_filtered_features
    y_labels = labels

    classification_type = 'CL5Levels'
    #get train and test data and labels according to ratio and classification_type
    X_train, Y_train, X_test, Y_test =  get_train_test(X_data, y_labels, ratio, between_shuffle=between_shuffle, classification_type=classification_type)
    #train several models
    dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = no_classifiers, classification_type=classification_type)
    print('finish train for CL5Levels')
    # Display results
    display_dict_models(dict_models)



if __name__ == '__main__':


    #####################################################################################
    ###
    ###             process data and labels to reduce dimensions
    ######################################################################################    #understand  relevant features and update X accordingly
    X_filtered_features = select_features(x_input, y_for_filtering)

    #If we want another features different for conditions prediction
    # y_for_filtering =
    # X_filtered_features_st = select_features(x_input, y_for_filtering)

    # # Significant columns to use in tested data
    significant_features = X_filtered_features.columns
    num_significant_features = len(significant_features)

    to_filter_by = from_columns(X_filtered_features) # dictionary
    pickle.dump(to_filter_by, open("saved_features_no_PCA.p", "wb"))  # save it into a file named saved_features.p

    # Load the dictionary back from the pickle file.
    # features_loaded = pickle.load(open("saved_features.p", "rb"))

###################################################
    try_pca_n_component = 15
    #create PCA
    pca_n_component = min(try_pca_n_component, num_significant_features)
    pca_n_component
    pca_allData = PCAForPandas(n_components= pca_n_component)

    X_pca = pca_allData.fit_transform(X_filtered_features) #YONIT
    # save pca into a file named pca_allData.p
    with open("saved_pca.pickle", "wb") as file_:
        pickle.dump(pca_allData, file_, -1)

    # #fit PCA on training set
    # pca.fit(X_filtered_features)
    # pca.transform(X_filtered_features)

    X_data = X_filtered_features


    #load pca FOR MAINBLACK
    # pca_loaded = pickle.load(open("saved_pca.p", "rb"))
    # pca_loaded.transform(data)

    # if PCA:
    #     # pca_allData = PCAForPandas(n_components=50)
    #     pca = PCA(copy=True, iterated_power='auto', n_component=pca_n_component, random_state=None,
    #               svd_solver='auto', tol=0.0, whiten=False)
    #     pca.fit(df_features_loaded)
    #     columns = ['pca_%i' % i for i in range(pca_n_component)]
    #     df_pca = DataFrame(pca.transform(df_features_loaded), columns=columns, index=df_features_loaded.index)
    #     pca.fit(df_features_loaded)



    # pickle.dump(to_filter_by, open("saved_features_PCA.p", "wb"))  # save it into a file named saved_features.p
    #


    # test_window = pd.read_csv(r'C:\Users\User\Documents\limudim\project\python_code\Test\one_sample_raw\51.csv')
    # test_input = extract_features(test_window, column_id='sub_id', column_sort='time',
    #                               kind_to_fc_parameters=to_filter_by)

    # X_pca = X_filtered_features
    # X_pca.to_csv(r'C:\Users\User\Documents\limudim\project\python_code\rawDataAndFeatures\csv_per_sub\02_raw_data_subjects\filtered data generalized\pca_allDATA.csv')


    # X_filtered_features = select_features(X, y_for_filtering)
    # # X.to_csv(r'C:\\python_code\full_flow\filtered_features_all_data.csv') - this should be loaded in gui script once not work
    #
    # significant_features = X_filtered_features.columns  # this are the relevant features
    # print(len(significant_features))  # for test
    # to_filter_by = from_columns(X_filtered_features)  # dict for test
    #
    # X = X_filtered_features  # rename
    # # #PCA on data
    # pca_allData = PCAForPandas(n_components=5)
    # X_pca = pca_allData.fit_transform(X)
    #
    # # X_pca = X_filtered_features # to run with no pca
    # # X_pca.to_csv(r'C:\Users\User\Documents\limudim\project\python_code\rawDataAndFeatures\csv_per_sub\02_raw_data_subjects\filtered data generalized\pca_allDATA.csv')
    #
    # test_window = pd.read_csv(r'C:\Users\User\Documents\limudim\project\python_code\Test\one_sample_raw\51.csv')
    # test_input = extract_features(test_window, column_id='sub_id', column_sort='time',
    #                               kind_to_fc_parameters=to_filter_by)
    # print(len(test_input.columns))
    # test_input = test_input[significant_features]
    # print(len(test_input.columns))
    # test_pca = pca_allData.transform(test_input)
    classify()
    # classify()


# clasification_type = ["CL5Levels", "CL2levels", "CL2Condition2", "condition2"]
