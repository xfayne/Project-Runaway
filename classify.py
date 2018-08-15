from scipy.fftpack import fft
from scipy.signal import welch
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from siml.sk_utils import *
# from siml.signal_analysis_utils import *

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import load_data as ld

ld.path
ld.TRAIN_INPUT
ld.TRAIN_LABELS
ld.TEST_LABELS
ld.TEST_INPUT

type_input = 2

train_signals, test_signals, train_labels, test_labels = ld.load_data_mat(ld.path, 0, ld.type_input)


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
   ----------
   x : 1D array_like
       data.
   mph : {None, number}, optional (default = None)
       detect peaks that are greater than minimum peak height.
   mpd : positive integer, optional (default = 1)
       detect peaks that are at least separated by minimum peak distance (in
       number of data).
   threshold : positive number, optional (default = 0)
       detect peaks (valleys) that are greater (smaller) than `threshold`
       in relation to their immediate neighbors.
   edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
       for a flat peak, keep only the rising edge ('rising'), only the
       falling edge ('falling'), both edges ('both'), or don't detect a
       flat peak (None).
   kpsh : bool, optional (default = False)
       keep peaks with same height even if they are closer than `mpd`.
   valley : bool, optional (default = False)
       if True (1), detect valleys (local minima) instead of peaks.
   show : bool, optional (default = False)
       if True (1), plot data in matplotlib figure.
   ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
   -------
   ind : 1D array_like
       indeces of the peaks in `x`.
    Notes
   -----
   The detection of valleys instead of peaks is performed internally by simply
   negating the data: `ind_valleys = detect_peaks(-x)`

   The function can handle NaN's """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    # if show:
    #     if indnan.size:
    #         x[indnan] = np.nan
    #     if valley:
    #         x = -x
    #     _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind




##############################################################
#2.1 FFT
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values



def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values
###############################################

sample_rate = 1;

def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [sample_rate * kk for kk in range(0, len(y_values))]
    return x_values, y_values


####

labels = ['x-component', 'y-component', 'z-component']
colors = ['r', 'g', 'b']
suptitle = "Different signals for the activity: {}"

xlabels = ['Time [sec]', 'Freq [Hz]', 'Freq [Hz]', 'Time lag [s]']
ylabel = 'Amplitude'
axtitles = [['Acceleration', 'Gyro', 'Total acceleration'],
            ['FFT acc', 'FFT gyro', 'FFT total acc'],
            ['PSD acc', 'PSD gyro', 'PSD total acc'],
            ['Autocorr acc', 'Autocorr gyro', 'Autocorr total acc']
            ]

list_functions = [get_values, get_fft_values, get_psd_values, get_autocorr_values]



t_n = 8.80 #to
N = 2200 #number of sampling
T = t_n / N #sampling rate?
f_s = 1/T #per seconds?

signal_no = 0
signals = train_signals[signal_no, :, :]
label = train_labels[signal_no]
# activity_name = activities_description[label]

# f, axarr = plt.subplots(nrows=4, ncols=3, figsize=(12, 12))
# f.suptitle(suptitle.format(activity_name), fontsize=16)

# for row_no in range(0, 4):
#     for comp_no in range(0, 9):
#         col_no = comp_no // 3
#         plot_no = comp_no % 3
#         color = colors[plot_no]
#         label = labels[plot_no]
#
#         axtitle = axtitles[row_no][col_no]
#         xlabel = xlabels[row_no]
#         value_retriever = list_functions[row_no]
#
#         ax = axarr[row_no, col_no]
#         ax.set_title(axtitle, fontsize=16)
#         ax.set_xlabel(xlabel, fontsize=16)
#         if col_no == 0:
#             ax.set_ylabel(ylabel, fontsize=16)
#
#         signal_component = signals[:, comp_no]
#         x_values, y_values = value_retriever(signal_component, T, N, f_s)
#         ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
#         if row_no & gt; 0:
#             max_peak_height = 0.1 * np.nanmax(y_values)
#             indices_peaks = detect_peaks(y_values, mph=max_peak_height)
#             ax.scatter(x_values[indices_peaks], y_values[indices_peaks], c=color, marker='*', s=60)
#         if col_no == 2:
#             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.subplots_adjust(top=0.90, hspace=0.6)
# plt.show()

def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks


def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y


def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0, dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]

            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100 - percentile)
            # ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min) / denominator

            features += get_features(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features(*get_fft_values(signal, T, N, f_s), mph)
            features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)


denominator = 10
X_train, Y_train = extract_features_labels(train_signals, train_labels, T, N, f_s, denominator)
X_test, Y_test = extract_features_labels(test_signals, test_labels, T, N, f_s, denominator)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
print("Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
Y_test_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_test_pred))
