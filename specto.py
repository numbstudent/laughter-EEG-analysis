from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pickle
from sklearn.decomposition import FastICA

# filename=['gav','ovi','ila','hrz','aldoh','cra','cips','vyn','rijuu','sinlo','skot','manai','nature']
filename=['ovi']
# filename=[    'cra','cips','vyn','rijuu','sinlo','skot','manai','nature']
useica = 0
X1 = []
X2 = []
# for ww in range(0, len(filename)):
for ww in range(0, len(filename)):
    with open('Z:/nani/experiment/'+filename[ww]+'/funnylaugh_yes.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        d1 = pickle.load(f)
    with open('Z:/nani/experiment/'+filename[ww]+'/funnylaugh_no.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        d2 = pickle.load(f)
    if ww==0:
        X1 = d1[0]
        X2 = d2[0]
    else:
        X1 = np.append(X1,(d1[0]),axis=0)
        X2 = np.append(X2,(d2[0]),axis=0)
X1 = np.array(X1)
X2 = np.array(X2)

data = X2
data = np.mean(data,axis=0)
channel = 5

if useica==1:
    ica = FastICA(n_components=14, max_iter=500)
    data = ica.fit_transform(data)

x = data[:,channel]

fs = 128.

f, t, Sxx = signal.spectrogram(x, fs, window="hamming",  nfft=256, scaling="density")
# t = [0,5]
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from obspy.core import read
# from obspy.signal.tf_misfit import cwt
# import pylab
#
# # tr = read("whole.sac")[0]
# tr = x
# npts = 128
# dt = 5
# t = np.linspace(0, 640, 640)
# print(t.shape)
# f_min = 1
# f_max = 50
#
# scalogram = cwt(tr, dt, 8, f_min, f_max)
#
# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
# ax2 = fig.add_axes([0.1, 0.75, 0.75, 0.2])
# ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
# img = ax1.imshow(np.abs(scalogram)[-1::-1], extent=[t[0], t[-1], f_min, f_max],
#           aspect='auto', interpolation="nearest")
#
# ax1.set_xlabel("Time after %s [s]" % 0)
# ax1.set_ylabel("Frequency [Hz]")
# ax1.set_yscale('linear')
# ax2.plot(t, tr, 'k')
# pylab.xlim([30,72])
#
# fig.colorbar(img, cax=ax3)
#
# plt.show()
