import sys, os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication,QDialog, QFileDialog
from PyQt5.uic import loadUi
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.io as sio
import threading, subprocess
from random import randint
import sched, time as Time
import matlab.engine
import pickle

#pg.setConfigOption('background', 'w')
#pg.setConfigOption('foreground', 'k')

count1 = 0
FILEPATH = os.path.abspath(__file__)

NETWORK = 0 #NETWORK == 1 <=> we use neural network, NETWORK == 0 <=> we usr classify

ECGdata = []
GSRdata = []
data = [GSRdata,ECGdata]

ECGtime = []
GSRtime = []
time = [GSRtime,ECGtime]

results = [1,3,2,4,1,3]

filename = 'finalized Gradient Boosting Cllasifier.sav'

########## methods for real time grahps plotting ############
# Run .mat scripts
eng = matlab.engine.start_matlab()  ##Need to bring up

def predictSeg(ECGsignal, GSRsignal):

    if NETWORK == 1:
        input_seg = eng.mat_signals(ECGsignal, GSRsignal)  # Run .mat script to filter signals
        X = input_seg
        return input_seg
    else:
        input_features = eng.mat_features(ECGsignal, GSRsignal)  # Run .mat script to extruct features from signals

        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(X_test, Y_test)

        # predict the result of the segment
        result = loaded_model.predict(input_features)
        print(result)
        return result



def updateGSRplot(starttime, update_time):
    update_time = 0.2
    buf_read = 200
    # updates the raw GSR plot
    # getting paramater 'starttime' = time of the clock when this method had been called
    # every update_time= 0.2 sec the plot is being updating with more 200 data value
    global GSRdata, GSRtime,GSRcurve,count1
    RTdata = []
    RTtime = []
    while True:
        #The simulation: reading the data in real time
        RTdata.extend(GSRdata[count1:(count1+buf_read)])
        RTtime.extend(GSRtime[count1:(count1+buf_read)])
        count1+=buf_read
        print(RTdata[count1-1]) #Printing the data in the kernel
        GSRcurve.setData(RTtime,RTdata) #Updates the gsr plot
        QtGui.QApplication.processEvents()
        Time.sleep(update_time - ((Time.time() - starttime) % update_time)) #Time.sleep(x) hold the system for x time

'''def updateGSRplot(starttime):
    # updates the raw GSR plot
    # getting paramater 'starttime' = time of the clock when this method had been called
    # every 0.2 sec the plot is being updating with more 200 data value
    global GSRdata, GSRtime,GSRcurve,count1
    while True:
        count1+=200
        print(GSRdata[count1])
        GSRcurve.setData(GSRtime[0:count1],GSRdata[0:count1])
        QtGui.QApplication.processEvents()
        Time.sleep(0.2 - ((Time.time() - starttime) % 0.2)) #Time.sleep(x) hold the system for x time'''

    
def updateECGplot(starttime,sync_time):
    # updates the raw ECG plot
    # getting paramater 'starttime' = time of the clock when this method had been called
    # and aramater 'sync_time' = time of the GSR when this method had been called
    # every 0.2 sec the plot is being updating with more 200 data value
    global ECGdata ,ECGtime, ECGcurve
    count2 = sync_time
    while True:
        count2+=200
        print(ECGdata[count2])
        ECGcurve.setData(ECGtime[sync_time:count2],ECGdata[sync_time:count2])
        QtGui.QApplication.processEvents()
        Time.sleep(0.2 - ((Time.time() - starttime) % 0.2))

    
############## methods for gui interfence ###################

#Main for GUI
class runBlack (QDialog):
    def __init__(self):
        super(runBlack,self).__init__()
        loadUi('runBlack.ui',self) #User interface
        self.setWindowTitle('hello world')
        self.loadGSR.clicked.connect(self.loadGSR_clicked)
        self.loadECG.clicked.connect(self.loadECG_clicked)
        self.start.clicked.connect(self.start_clicked)
        self.reset.clicked.connect(self.reset_clicked)


    @pyqtSlot()
    
    
    def loadECG_clicked(self):
         
        #self.rawDataView.clear()
        global ECGdata, data, GSRdata, ECGtime, GSRtime, time, ECGcurve, start_time
        fecg = QFileDialog.getOpenFileName(self, 'Open file',"mat files (.mat)")[0]
        '''m= sio.loadmat(fecg)
        tmp = m['time_index'][0]
        ECGtime = []
        for t in tmp: ECGtime += [t/1000]
        ECGdata = m['data'][0]
        del m'''
        m = sio.loadmat(fecg)
        ECGtime = m['times'][0]
        ECGdata = m['data'][0]

        #ECGdata = np.random.normal(loc=6.5, scale=0.3, size=len(GSRtime)) #simulates graph
        data = [GSRdata,ECGdata]
        time = [GSRtime,ECGtime]
        
        if self.rtMode.isChecked() == True:
        # load ECG clicked when the system is on real time mode #
            sync_time = count1 # = GSR current position
            start_time=Time.time() #Current time
            threading.Timer(0, updateECGplot, (start_time,sync_time,)).start() #start ploting ECG on new thread
            
        else:
        # load ECG clicked when the system is on static test mode #
            ECGcurve.setData(ECGtime,ECGdata)

    #GSR#####################
    def loadGSR_clicked(self):
        
        global ECGdata, data, GSRdata, ECGtime, GSRtime, time, GSRcurve, ECGcurve, p
        
        if GSRdata == []: #If empty
            #Initialize plot
            p = self.rawDataView.addPlot()
            ECGcurve = p.plot(pen=(1,2))
            GSRcurve = p.plot(pen=(0,2))

        #Load only .mat file
        fgsr = QFileDialog.getOpenFileName(self, 'Open file',"mat files (.mat)")[0]
        m = sio.loadmat(fgsr)
        GSRtime = m['time_index'][0]
        GSRdata = m['data'][0]
        del m #We don't need the matrix anymomre
        data = [GSRdata,ECGdata]
        time = [GSRtime,ECGtime]

        #Positions in window
        p.setXRange(-1.06, 878.15)
        p.setYRange(4.82, 9.3)
        
        if self.rtMode.isChecked() == True:
        # load ECG clicked when the system is on real time mode #
            starttime=Time.time() #Current time
            threading.Timer(0, updateGSRplot, (starttime,)).start() #start ploting GSR on new thread

        else:
        # load GSR clicked when the system is on static test mode #
            GSRcurve.setData(GSRtime,GSRdata)


                
    def start_clicked(self):
        
        global data, ECGdata, GSRdata
        
        if ((ECGdata!= []) and (GSRdata!= [])):
        # both GSR & ECG files has been loaded
            if self.rtMode.isChecked() == True:
            # system is on real time test mode #
                data = [1,3,5,2,6,2]
                self.rawDataView.plot([0,1,2,3,4,5],data,pen=(0,2))
            
            else:
            # system is on static test mode #
                data = [GSRdata,ECGdata]
                p1 = self.modDataView.addPlot(x=time[1], y=data[1], pen=(1,2))
                p2 = pg.PlotCurveItem(x=time[0], y=data[0], pen=(0,2))
                p1.addItem(p2)


                #Add classification








                #After we predict all the blocks levels
                for i in range (len(results)):
                # add the network workeloads predictons of each data block to the graphic view
                    level = results[i]
                    lr = pg.LinearRegionItem(values=[150*(i), 150*(i+1)], brush=pg.intColor(index=level,alpha=50), movable=False)
                    p1.addItem(lr)
                    label = pg.InfLineLabel(lr.lines[1], "oveload "+str(level) , position=0.85, rotateAxis=(1,0), anchor=(1, 1))

        else:
        # either GSR or ECG files has not been loaded
            print("error")
    
    def reset_clicked(self):
        try:
            subprocess.Popen([sys.executable, FILEPATH])
        except OSError as exception:
            print('ERROR: could not restart aplication:')
            print('  %s' % str(exception))
        else:
            QApplication.quit()
        
    
app=QApplication(sys.argv)
widget = runBlack()
widget.show()
sys.exit(app.exec_())
