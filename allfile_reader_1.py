import numpy as np
import seaborn as sns
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
loc = 'Z:/nani/experiment/skot/short laugh 1/short laugh 1_2019.06.12_17.18.23.csv'
data = genfromtxt(loc, skip_header=1, delimiter=',')
df = pd.read_csv(loc, header=None, skiprows=1)
df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
peteek = (df.query('t == "100"').index)
startpoint = (df.query('t == "100"').index[0])
# startpoint = 0
data = data[:,2:16]
print(data.shape)
useica = 0
multivalue = 200
if useica==1:
    ica = FastICA(n_components=14, max_iter=500)
    data = ica.fit_transform(data)
    multivalue /= 10000
for x in range(0,14):
    data[:,x] = data[:,x] + (multivalue*x)
mbohtalah = startpoint
ranges = [mbohtalah]
mbohtalah = mbohtalah+(11.2*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(30.530*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(51.318*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(70.361*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(89.844*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(110.016*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(129.214*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(148.051*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(167.716*128)
ranges.append(mbohtalah)
mbohtalah = mbohtalah+(185.891*128)
ranges.append(mbohtalah)
ds1 = []
ds2 = []
for z in range(0,len(ranges)-1):
    ds1.append(data[int(round(ranges[z]))-(3*128):int(round(ranges[z+1]))+(10*128)])
ds2=ds1
print(np.array(ds1).shape)
print(np.array(ds2).shape)
# exit()
# for mbuoh in range(0,20):
#     if mbuoh%4 ==0:
#         plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='g', alpha=0.5,zorder=1)
#     elif mbuoh%4 ==1:
#         plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='b', alpha=0.5,zorder=1)
#     elif mbuoh%4 ==2:
#         plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='r', alpha=0.5,zorder=1)
#     else:
#         plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='b', alpha=0.5,zorder=1)
# # plt.axvspan(peteek[0], peteek[1], facecolor='y', alpha=0.5,zorder=1)
# plt.axvspan(0, peteek[1]-peteek[0], facecolor='y', alpha=0.5,zorder=1)
# plt.plot(data[peteek[0]:],lw=0.2)
# plt.show()

# plt.plot(np.array(ds1).reshape((3200,14)),lw=0.5, color="k")
# plt.plot(np.array(ds2).reshape((3200,14)),lw=0.5, color="b")
# plt.axvline(x=640, color="r")
# plt.axvline(x=1280, color="r")
# plt.axvline(x=1920, color="r")
# plt.axvline(x=2560, color="r")
# plt.show()
#
# dsa = np.mean(ds1,axis=0)
# dsb = np.mean(ds2,axis=0)
# plt.plot(dsa, color="k")
# plt.plot(dsb, color="b")
# plt.show()

####################FFT,WELCH########################
import numpy as np
# data = np.array(ds1).reshape((3200,14))[:,9]
# data2 = np.array(ds2).reshape((3200,14))[:,9]
# data = np.array(ds1)[1,:,9]
# data2 = np.array(ds1)[2,:,9]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

# Define sampling frequency and time vector
sf = 128.

# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# for ww in range(0,5):
#     data = np.array(ds1)[ww,:,4]
#     data = data * np.hamming(640)
#     time = np.arange(data.size) / sf
#     plt.plot(time, data, lw=1.5, color='k', alpha=0.5)
data = np.mean(ds1,axis=0)[:,0]
# data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
time = np.arange(data.size) / sf
plt.plot(time, data, lw=1.5, color='k', alpha=0.5)
# for ww in range(0,5):
#     data = np.array(ds2)[ww,:,4]
#     data = data * np.hamming(640)
#     time = np.arange(data.size) / sf
#     plt.plot(time, data, lw=1.5, color='b', alpha=0.5)
data = np.mean(ds2,axis=0)[:,0]
# data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
time = np.arange(data.size) / sf
plt.plot(time, data, lw=1.5, color='b', alpha=0.5)
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.xlim([time.min(), time.max()])
plt.title('N3 sleep EEG data (9)')
sns.despine()
plt.show()

from scipy import signal

# Define window length (4 seconds)
win = 4 * sf

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
# for ww in range(0,5):
#     data = np.array(ds1)[ww,:,4]
#     freqs, psd = signal.welch(data, sf, nperseg=win)
#     # psd = psd*psd
#     plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
data = np.mean(ds1,axis=0)[:,0]
# data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
freqs, psd = signal.welch(data, sf, nperseg=win)
plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
# for ww in range(0,5):
#     data = np.array(ds2)[ww,:,4]
#     freqs, psd = signal.welch(data, sf, nperseg=win)
#     # psd = psd*psd
#     plt.plot(freqs, psd, color='b', lw=1, alpha=0.5)
data = np.mean(ds2,axis=0)[:,0]
# data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
freqs, psd = signal.welch(data, sf, nperseg=win)
plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
# plt.ylim([0, 2500])
plt.title("Welch's periodogram")
plt.xlim([0, freqs.max()])
sns.despine()
plt.show()

exit()
#####################################################
# plt.figure(figsize=(64,64))
# plt.xticks(rotation=90)

data = np.cov(np.transpose(dsa))
sns.heatmap(data)
plt.show()
data = np.cov(np.transpose(dsb))
sns.heatmap(data)
plt.show()
exit()
data = np.cov(np.transpose(ds1[0]))
sns.heatmap(data)
plt.show()
data = np.cov(np.transpose(ds1[1]))
sns.heatmap(data)
plt.show()
data = np.cov(np.transpose(ds1[2]))
sns.heatmap(data)
plt.show()
data = np.cov(np.transpose(ds1[3]))
sns.heatmap(data)
plt.show()
data = np.cov(np.transpose(ds1[4]))
sns.heatmap(data)
plt.show()
