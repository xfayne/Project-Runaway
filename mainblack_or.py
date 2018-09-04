import pandas as pd
import sys, os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.io as sio
from PyQt5.QtGui import QIcon, QPixmap
import threading, subprocess
# from ml_process import *
from random import randint
from tsfresh.utilities.dataframe_functions import impute
import sched, time as Time
import pickle
from subprocess import Popen, PIPE
from tsfresh.feature_extraction.settings import from_columns, ComprehensiveFCParameters
from tsfresh import extract_features, extract_relevant_features, select_features
from pandas import DataFrame
#import matlab.engine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')


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

FILEPATH = os.path.abspath(__file__)

ECGdata = []
GSRdata = []

ECG_cleard_data = []
GSR_cleard_data = []


eventsData = []
eventsTimes = []

blocks = []

data = [GSRdata, ECGdata]

ECGtime = []
GSRtime = []

blockStart = blockEnd = 0




##RTStream
#For simulting Real Time data
#We load the stream file here:
RTStreamTIME = np.array([])
RTStreamECG = np.array([])
RTStreamGSR = np.array([])
RTStreamEVENTS = np.array([])

#Pointer to the current data streamed
#Index in: RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS
pointer = 0

#Events
start_block = 1 #'run_start'
end_block = 2 #'run_end'
end_data = 3 #'finish'
events = [start_block, end_block, end_data]

#Stream
#Put the data each sample here
#The data stream comes here: Data[0] = ecg val, Data[1] = gsr val,
#Data[2] = event/notEvent. 0: notEvent, 1: start block, 2: end block, 3: end data
DataStream = np.zeros(3)
#The time data stream
TimeStream = np.array([0])

#RTUpdateSignals
#The sequence data and time
RTtime = []
RTdata = [[], []]
##Pointer to the current data added from stream
#To sequence data
index = 0

#Predictions to current block
#Clear every time we read new block
blockPredictions = []

#Predictions to current block (stress/ noStress)
#Clear every time we read new block
blockPredictions_st = []

#Counter to number of different blocks reached so far
block_num = 1

#Counting till seg_len reading data in the same block
#Reset the counter every new segment reading
counter_seg = 0

#Flag: if we reading data from block or not
reading_block = 0

times = [GSRtime, ECGtime]

time_cleard = []


#ML
NETWORK = 0 #Use cllasifier or neural network
filename = 'finalized Gradient Boosting Classifier.sav'
filename_st = 'st_finalized Gradient Boosting Classifier.sav'

# load the model from disk: CL
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

# load the model from disk: Conditions
loaded_model_st = pickle.load(open(filename_st, 'rb'))

#window size
seg_len = 2200


#Features fo classify: (CL, Conditions)
#Significant features to extracet
#Load dictionary with the significant features
dic_features_loaded = pickle.load(open("saved_features_no_PCA.p", "rb"))
#Load names of significant features for BUG solving
columns_features_loaded = pickle.load(open("saved_filtered_features_columns.sav", "rb")) #yonit
#Make data frame from dictionary
df_features_loaded = pd.DataFrame.from_dict(dic_features_loaded)
#PCA is FINE?********

#PCA on data
PCA = False
pca_n_component = 50

if PCA:
    # Load pca that can be used if pca = TRUR - YONIT
    pca_loaded = pickle.load(open("saved_pca.pickle", "rb", -1))


# # X_pca = features_loaded
#
# X_pca = pca_allData.fit_transform(X_filtered_features)
# print(X_pca)

# to_filter_by = from_columns(X_pca)  # dictionary

# pickle.dump(to_filter_by, open("saved_features_PCA.p", "wb"))  # save it into a file named saved_features.p


ecg_features = []
gsr_features = []

tags_times = []

global to_filter_by

# #Make dictionary of relavent features to extract from raw signal
# filtered_relevant_features = []
# dict_parameters = from_columns(filtered_relevant_features)

##############################

import numpy

def smooth(x,window_len=5,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
        

    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    

    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
    
    y=numpy.convolve(w/w.sum(),s,mode='valid')
    #return y
    res = y[int((window_len/2))-1:-int((window_len/2))-1]
    #res = res[:-1]
    return res    

def clearSignal(signal):
    clearedSignal = []
    return signal


#################################
def RTupdate(self):
    #print('RTupdate')
    RTStream() #get current data: ecg, gsr and event and updates in DataStream
    RTupdateSignals(self)

########## methods for real time grahps plotting and predictions ############
#Need to set end reading correctly**********88

#Simulates Real Time streaming data
#Takes from global variables: RTStreamECG, RTStreamGSR, RTStreamEVENTS, RTStreamTIME, DataStream
#Put in DataStream the data
def RTStream():
    global RTStreamECG, RTStreamGSR, RTStreamEVENTS, RTStreamTIME, DataStream, TimeStream, end_data, pointer

    #Streaming new data
    np.put(DataStream, [0,1,2], [RTStreamECG[pointer], RTStreamGSR[pointer], RTStreamEVENTS[pointer]])
    np.put(TimeStream, 0, RTStreamTIME[pointer])

    #Finish reading all data
    if RTStreamEVENTS[pointer] == end_data:
        #finish = True
        print('DataStream: exit')

    #Update the pointer
    pointer += 1


#input: block of GSR snd ECG signals.
#output: median of all the seg_len size predictions in the block
def predictBlock(GSRsignal,ECGsignal):
    i = 0
    predictions = []
    while (i+seg_len < len(GSRsignal)):
        predict = predictSeg([ GSRsignal[i:i+seg_len],ECGsignal[i:i+seg_len] ])
        predictions.extend(predict)
        i+= seg_len
    return int(np.median(predictions))


#input: event data list and event samples time.
#output: list of block time in the session: [(block i Start,block i End)]
def makeBlocksList(tags, tags_time):
    lst = []
    for i in range(len(tags)):
        if (tags[i] == start_block): lst.append([tags_times[i], ])
        if (tags[i] == end_block): lst[-1].append(tags_times[i])
    return lst


#input: signal and flag specify the data type.
#flag = 0 <=> is ecg, flag = 1 <=> is gsr
#Excecute .mat script to extract features from signal flag: 0 if ecg. 1 if gsr
def extractSeg(signal, flag):
    print('extractSeg')


#Gets inputs and loaded model
#inputes : array's length = 3. inputes[0] = ecg, inputes[1] = gsr, inputes[2] = events
#seg_time: series of the times
#type_label: condition or CL
#loaded_model for prediction
#Predict Cl/ condition of segment and return result.
#The function excecute two threads for calculate features of gsr and ecg parallel
def predictSeg(inputs, seg_time, type_label):
    # global seg_len
    # #Check:'''''''''''''
    # filename_glass = r'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\data.csv'
    # df_data = pd.read_csv(filename_glass)
    #
    # # Take only levels in: levels
    # levels = [1, 2, 3, 4, 5]
    # df_data = df_data.loc[df_data['level'].isin([1,2,3,4,5])]
    # y_cols = 'level'
    # x_cols = list(df_data.columns.values)
    # x_cols.remove(y_cols)
    # x_cols.remove('sub_num')
    # x_cols.remove('conditions')
    # df_data = df_data[x_cols].values
    # features = df_data[0]

    #"""""""""""""""""""""""""""""""""""""""""""""""

    # seg_time = seg_len / 250
    # inputs[0] ECG
    # inputs[1] GSR
    # time
    # time = np.arange(0, seg_sec, 0.04)


    # arrayTitles = ['sub_id', 'time', 'ecg', 'gsr']
    # time = np.arange(0, seg_sec, 0.04)
    # dict_parameters = to_filter_by
    #dic_features_loaded
    #Arrange the input to extract features
    id = np.full((seg_len), 1)
    full_array = np.empty([seg_len, 4])
    full_array[:, 0] = id
    full_array[:, 1] = seg_time
    full_array[:, 2] = inputs[0] #ECG
    full_array[:, 3] = inputs[1] #GSR
    raw_input = pd.DataFrame(full_array, columns=['id', 'time', 'ecg', 'gsr'])
    # Extract features
    input_features = extract_features(raw_input, column_id='id', column_sort='time',
                                      kind_to_fc_parameters=dic_features_loaded)

    input_features_fixed = input_features[columns_features_loaded]  # yonit
    # PCA: (need to add)
    impute(input_features)
    input_features_fixed = input_features[columns_features_loaded]  # yonit
    result = -1
    if PCA: #yonit
        input_pca = pca_loaded.transform(input_features_fixed) #yonit
        if type_label == 'CL':

            result = loaded_model.predict(input_pca.values) #yonit
        elif type_label == 'condition':
            result = loaded_model_st.predict(input_pca.values)  # yonit
    # input_pca = input_features
    # input_pca = input_features
    # features = input_pca.values
    # predict the result of the segment
    # features = np.asarray(features)
    # features = features.reshape(1, -1)  #Maybe need
    # result = loaded_model.predict(input_features)
    else: #yonit
        if type_label == 'CL':
            result = loaded_model.predict(input_features_fixed.values)  # yonit
        elif type_label == 'condition':
            result = loaded_model_st.predict(input_features_fixed.values)  # yonit
    print('predictSeg : result is:', result)  # For check

    return result


#Update in RT the signals plots and calling the predictSeg when needed
#Calculate the final predict of the block
def RTupdateSignals(self):

    global DataStream, TimeStream, index, counter_seg, reading_block, RTtime, RTdata, blockPredictions, block_num,time_cleard
    global ECGcurve, ECGcurveClear, GSRcurve, GSRcurveClear,ECG_cleard_data,GSR_cleard_data, cleardPlot, blockStart, blockEnd, p1, m1

    #When we update the plots of the data streaming
    display_jump = 10

    #flag end of streaming data
    # finish = False

    # Insert to RTData array ecg and gsr values from the data stream
    # RTdata[0].append(DataStream[0])
    # if(DataStream[1]!=0):RTdata[1].append(DataStream[1])
    # else: RTdata[1].append(5.96)
    RTdata[0].append(DataStream[0])
    RTdata[1].append(DataStream[1])
    RTtime.append(TimeStream[0])

    if counter_seg == seg_len: #Finish reading segment
        print('Finish reading segment from block number:', block_num)
        ECGseg = RTdata[0][index-counter_seg+1:index+1]
        GSRseg = RTdata[1][index-counter_seg+1:index+1]
        seg_times = RTtime[index-counter_seg+1:index+1]

        #Clear the seg signals we read
        GSRsegClear = smooth(np.asarray(GSRseg))
        ECGsegClear = clearSignal(ECGseg)

        #Updating the proceesed data arrays for displays later
        ECG_cleard_data.extend(ECGsegClear)
        GSR_cleard_data.extend(GSRsegClear)
        time_cleard.extend(seg_times)

        # #like it was before, but now with the cleared signal
        #segment = [GSRsegClear,ECGsegClear]
        segment = [ECGseg, GSRseg]
        #Predict the segment
        cl_result = predictSeg(segment, seg_times, 'CL')
        st_result = predictSeg(segment, seg_times, 'condition')
        #segment = [RTdata[0][index-counter_seg+1:index+1],RTdata[1][index-counter_seg+1:index+1]]
        print('segment', type(segment))
        #@Real code
        self.lcdNumber.display(cl_result)
        self.progressBar.setValue(cl_result)
        #graphic display of thr segments prediction
        blockPredictions.append(cl_result) #Add predict to list of segmentations' predicts in the block
        if st_result == 0: #no stress
            pixmap = QPixmap('no_stress.png')
            stress_scale = self.stress_level.setPixmap(pixmap)
            self.stress_level.resize(pixmap.width(), pixmap.height())
        elif st_result == 1: #stress
            pixmap = QPixmap('stress.png')
            stress_scale = self.stress_level.setPixmap(pixmap)
            self.stress_level.resize(pixmap.width(), pixmap.height())
        print('The result for the segment is:',cl_result) #For check
        print('The condition for the segment is:', st_result)  # For check
        counter_seg = 0 #Reset counter



    # Finish reading block data.
    # Predict the block:
    if DataStream[2] == end_block:
        blockEnd = RTtime[index]
        pixmap = QPixmap('neutral.png')
        stress_scale = self.stress_level.setPixmap(pixmap)
        self.stress_level.resize(pixmap.width(), pixmap.height())
        # OR: add label specify it's end of the block
        print('Finish reading block data. Need to predict the block')
        print(blockStart)
        print("---------")
        print(blockEnd)
        block_predict = int(np.median(blockPredictions))
        print('Predict of the block',block_num,'is:',block_predict)

        block_num += 1  # count another block

        #@Real code
        print('from RTupdateSignals: Predict is: ', block_predict) #For check
        #Clear predictions
        blockPredictions.clear()
        #Reset counter
        lr = pg.LinearRegionItem(values=[blockStart, blockEnd], brush=pg.intColor(index=block_predict, alpha=50), movable=False)
        m1.addItem(lr)
        label = pg.InfLineLabel(lr.lines[1],text=("BLOCK number: ", str(block_num-1), " overload " + str(block_predict)), position=(0.85), rotateAxis=(1, 0),anchor=(1, 1))
        counter_seg = 0

        #Or:
        #Do something with predict: on the graph show
        #Display what we predict for the block

    elif DataStream[2] == start_block: #Starting reading new block's data
        blockStart = RTtime[index] #TimeStream

        print(blockStart)
        
        #OR: add label specify it's start of block
        reading_block = 1 #Change flag

    if  reading_block == 1:

        #Update the counter_seg
        counter_seg += 1

    #Display plot
    if index % display_jump == 0 and index > 0:
        ECGcurve.setData(np.asarray(RTtime), np.asarray(RTdata[0]))
        GSRcurve.setData(np.asarray(RTtime), np.asarray(RTdata[1]))
        ECGcurveClear.setData(time_cleard,ECG_cleard_data)
        GSRcurveClear.setData(time_cleard,GSR_cleard_data)
        QtGui.QApplication.processEvents()

    if DataStream[2] == end_data:
        print('finish') #check
        #finish = True

    # Update index
    index += 1
#''Need to add return?

############## methods for GUI interfence ###################

def updateViews():
    ## view has resized; update auxiliary views to match
    global p1, p2, p3
    p2.setGeometry(p1.vb.sceneBoundingRect())    
    ## need to re-update linked axes since this was called
    ## incorrectly while views had different shapes.
    ## (probably this should be handled in ViewBox.resizeEvent)
    p2.linkedViewChanged(p1.vb, p2.XAxis)
    
def updateViewsMod():
    ## view has resized; update auxiliary views to match
    global m1, m2
    m2.setGeometry(m1.vb.sceneBoundingRect())    
    ## need to re-update linked axes since this was called
    ## incorrectly while views had different shapes.
    ## (probably this should be handled in ViewBox.resizeEvent)
    m2.linkedViewChanged(m1.vb, m2.XAxis)



class runBlack(QDialog):
    def __init__(self):
        global ECGcurve, GSRcurve, ECGcurveClear,GSRcurveClear, p1 ,p2, m1, m2
        global stress_scale
        super(runBlack, self).__init__()
        loadUi('runBlack.ui', self)
        self.setWindowTitle('CL Stress/No Stress Predictor')
        self.loadGSR.clicked.connect(self.loadGSR_clicked)
        self.loadECG.clicked.connect(self.loadECG_clicked)
        self.start.clicked.connect(self.start_clicked)
        self.reset.clicked.connect(self.reset_clicked)
        self.loadEvents.clicked.connect(self.loadEvents_clicked)
        self.RTinterface.clicked.connect(self.RTinterface_clicked)

        # Create widget
        # label = QLabel(self)

        pixmap = QPixmap('neutral.png')
        stress_scale = self.stress_level.setPixmap(pixmap)
        self.stress_level.resize(pixmap.width(), pixmap.height())

        ####### init raw data window #######
        ######OR
        pw = self.rawDataView
        pw.show()
        p1 = pw.plotItem
        p1.setLabels(left='GSR')
        p1.getAxis('left').setLabel(left='GSR', color='#ff0000')
        p1.setLabel('bottom', 'Time', units='s')
        ## create a new ViewBox, link the right axis to its coordinate system
        p2 = pg.ViewBox()
        p1.showAxis('right')
        p1.scene().addItem(p2) #scene?
        p1.getAxis('right').linkToView(p2)
        p2.setXLink(p1)
        p1.getAxis('right').setLabel('ECG', color='#00ffff')
        updateViews()
        p1.vb.sigResized.connect(updateViews)
        GSRcurve = p1.plot(pen=(0, 2))
        ECGcurve = pg.PlotCurveItem(pen=(1, 2))
        p2.addItem(ECGcurve)

        ####### init mod data window #######
        mw = self.modDataView
        mw.show()
        m1 = mw.plotItem
        m1.setLabels(left='GSR')
        m1.getAxis('left').setLabel(left='GSR', color='#ff0000')
        m1.setLabel('bottom', 'Time', units='s')
        ## create a new ViewBox, link the right axis to its coordinate system
        m2 = pg.ViewBox()
        m1.showAxis('right')
        m1.scene().addItem(m2)
        m1.getAxis('right').linkToView(m2)
        m2.setXLink(m1)
        m1.getAxis('right').setLabel('ECG', color='#00ffff')
        updateViewsMod()
        m1.vb.sigResized.connect(updateViewsMod)
        GSRcurveClear = m1.plot(pen=(0, 2))
        ECGcurveClear = pg.PlotCurveItem(pen=(1, 2))
        m2.addItem(ECGcurveClear)


        # p2 = self.rawDataView.addPlot(row=1, col=1, rowspan=1, colspan=1)
        # p = self.rawDataView.addPlot(row=50, col=1, rowspan=1, colspan=1)
        # ECGcurve = p2.plot(pen=(1, 2))
        # GSRcurve = p.plot(pen=(0, 2))
        # p.setLabel('right', 'Value', units='V')
        # p2.setLabel('right', 'Value', units='V')
        # p.setLabel('bottom', 'Time', units='s')
        # p2.setLabel('bottom', 'Time', units='s')
        #
        # p.setXRange(0, 5000)
        # p.setYRange(-20, 20)
        # p2.setXRange(0, 5000)
        # p2.setYRange(-600, 600)
        #
        # p.enableAutoScale()
        # p2.enableAutoScale()

    @pyqtSlot()


#Real Time mode: load data
    def RTinterface_clicked(self):
        if self.rtMode.isChecked() == True:

            #global ECGdata, data, GSRdata, ECGtime, GSRtime, time, GSRcurve, ECGcurve, p
            global ECGcurve, GSRcurve
            global RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS
            fstream = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
            m = sio.loadmat(fstream)
            RTStreamTIME = m['time_index'][13300:]
            RTStreamECG = m['ecg'][13300:]
            RTStreamGSR = m['gsr'][13300:]
            RTStreamEVENTS =  m['events'][13300:]
            # RTStreamTIME = m['time_index'][14000:]
            # RTStreamECG = m['ecg'][14000:]
            # RTStreamGSR = m['gsr'][14000:]
            # RTStreamEVENTS =  m['events'][14000:]            
            '''
            
            p = self.rawDataView.addPlot()

            # set properties
            # Need to change a bit the units scale
            # Need to make to windows for gsr and ecg seperate
            p.setLabel('left', 'Value', units='V')
            p.setLabel('right', 'Value', units='V')

            p.setLabel('bottom', 'Time', units='s')

            p.setXRange(0, 5000)
            p.setYRange(-600, 600)


            p.setWindowTitle('pyqtgraph plot')
            p.enableAutoScale()

            # plot
            ECGcurve = p.plot(pen='r')
            GSRcurve = p.plot(pen='b')
            # ECGcurve.getAxis("bottom").setLabel(text="Time", units="s")
            # ECGcurve.getYxis("right").setLabel(text="Time", units="s")'''


#######################################################

#Or:
#Static mode: load data
    def loadEvents_clicked(self): 
        
        global eventsData, eventsTimes, blocks
        blocks = []
        fevents = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m = sio.loadmat(fevents)
        eventsTimes = m['time_index'][0]
        eventsData = m['data'][0]
        for i in range(len(eventsData)):
            if (eventsData[i] == start_block): blocks.append([eventsTimes[i], ])
            if (eventsData[i] == end_block): blocks[-1].append(eventsTimes[i])
        del m
    
    def loadECG_clicked(self):

        # self.rawDataView.clear()
        global ECGdata, data, GSRdata, ECGtime, GSRtime, times, ECGcurve, start_time
        fecg = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m= sio.loadmat(fecg)
        ECGtime = m['time_index'][0]
        ECGdata = m['data'][0]
        del m
        '''
        ECGtime = GSRtime
        ECGdata = np.random.normal(loc=6.5, scale=0.3, size=len(GSRtime))'''
        data = [GSRdata, ECGdata]
        times = [GSRtime, ECGtime]
        ECGcurve.setData(ECGtime, ECGdata)


    def loadGSR_clicked(self):

        global ECGdata, data, GSRdata, ECGtime, GSRtime, times, GSRcurve, ECGcurve, p, p2
        fgsr = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m = sio.loadmat(fgsr)
        GSRtime = m['time_index'][0]
        GSRdata = m['data'][0]
        del m
        data = [GSRdata, ECGdata]
        times = [GSRtime, ECGtime]
        '''p.setXRange(-1.06, 878.15)
        p.setYRange(4.82, 9.3)'''
        GSRcurve.setData(GSRtime, GSRdata)

#######################################################

    #START
    def start_clicked(self):

        global data, ECGdata, GSRdata

        #RT variables for simulating streaming
        global RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS

        # Real Time mode
        if self.rtMode.isChecked() == True:
            # The data that will be used to simulate streaming is loaded
            #if ((RTStreamTIME != []) and (RTStreamECG != []) and (RTStreamGSR != []) and (RTStreamEVENTS != [])):
                print('started')
                self.timer = pg.QtCore.QTimer()
                self.timer.timeout.connect(lambda: RTupdate(self))
                self.timer.start(5)

        #Or:
        else:
            # system is on static test mode #
            if ((ECGdata!= []) and (GSRdata!= [])):
            # both GSR & ECG files has been loaded
                data = [GSRdata, ECGdata]
                p1 = self.modDataView.addPlot(x=times[1], y=data[1], pen=(1, 2))
                p2 = pg.PlotCurveItem(x=times[0], y=data[0], pen=(0, 2))
                p1.addItem(p2)
                for i in range(len(blocks)):
                #     # add the network workeloads predictons of each data block to the graphic view
                    blockStart = blocks[i][0]
                    blockEnd = blocks[i][1]
                    level = predictBlock(GSRdata[blocks[i][0]:blocks[i][1]], ECGdata[blocks[i][0]:blocks[i][1]])
                    lr = pg.LinearRegionItem(values=[blockStart, blockEnd], brush=pg.intColor(index=level, alpha=50), movable=False)
                    p1.addItem(lr)
                    label = pg.InfLineLabel(lr.lines[1], "oveload " + str(level), position=0.85, rotateAxis=(1, 0),anchor=(1, 1))
    
            else:
                # either GSR or ECG files has not been loaded
                 print('error')


    def reset_clicked(self):
        try:
            subprocess.Popen([sys.executable, FILEPATH])
        except OSError as exception:
            print('ERROR: could not restart aplication:')
            print('  %s' % str(exception))
        else:

            QApplication.quit()


#Start App
def run():
    app = QApplication(sys.argv)
    widget = runBlack()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
