import numpy as np
import seaborn as sns
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
# data = genfromtxt('Z:/nani/experiment/aldoh/dry laugh/dry laugh_2019.06.01_12.26.08.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/aldoh/funny laugh/funny laugh_2019.06.01_12.33.38.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/aldoh/short laugh 2/short laugh 2_2019.06.01_11.56.53.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/cra/funny laugh/funny laugh_2019.06.02_14.22.42.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/ila/dry laugh/dry laugh_2019.05.22_17.47.10.csv', skip_header=1, delimiter=',')
# loc = 'Z:/nani/experiment/ila/dry laugh/dry laugh_2019.05.22_17.47.10.csv'
# loc = 'Z:/nani/experiment/ovi/funny laugh/forced laugh funny_2019.05.22_12.32.52.csv'
# loc = 'Z:/nani/experiment/ila/funny laugh/forced funny laugh_2019.05.22_17.55.35.csv'
# loc = 'Z:/nani/experiment/aldoh/funny laugh/funny laugh_2019.06.01_12.33.38.csv'
# loc = 'Z:/nani/experiment/aldoh/dry laugh/dry laugh_2019.06.01_12.26.08.csv'
# loc = 'Z:/nani/experiment/cra/funny laugh/funny laugh_2019.06.02_14.22.42.csv'
# loc = 'Z:/nani/experiment/cra/dry laugh/dry laugh_2019.06.02_14.18.15.csv'
# loc = 'Z:/nani/experiment/rijuu/funny laugh/funny laugh_2019.06.07_15.51.39.csv'
# loc = 'Z:/nani/experiment/skot/funny laugh/funny laugh_2019.06.12_17.45.30.csv'
# loc = 'Z:/nani/experiment/skot/dry laugh/dry laugh_2019.06.12_17.40.45.csv'
# loc = 'Z:/nani/experiment/sinlo/funny laugh/funny laugh_2019.06.10_15.15.58.csv'
# loc = 'Z:/nani/experiment/sinlo/dry laugh/dry laugh_2019.06.10_15.12.31.csv'
# loc = 'Z:/nani/experiment/vyn/funny laugh/funny laugh_2019.06.05_13.46.41.csv'
# loc = 'Z:/nani/experiment/vyn/dry laugh/dry laugh_2019.06.05_13.39.19.csv'
# loc = 'Z:/nani/experiment/cips/funny laugh/funny laugh_2019.06.03_15.30.48.csv'
# loc = 'Z:/nani/experiment/cips/dry laugh/dry laugh_2019.06.03_15.26.57.csv'
# loc = 'Z:/nani/experiment/gav/funny laugh/forced funny laugh_2019.05.20_16.34.26.csv'
# loc = 'Z:/nani/experiment/gav/dry laugh/forced dry laugh_2019.05.20_16.29.02.csv'
# loc = 'Z:/nani/experiment/rot/funny laugh/forced laugh_2019.05.21_17.49.21.csv'
# loc = 'Z:/nani/experiment/rot/dry laugh/forced dry 2_2019.05.21_17.42.58.csv'
loc = 'Z:/nani/experiment/manai/funny laugh/funny laugh_2019.06.13_17.25.34.csv'
# loc = 'Z:/nani/experiment/manai/dry laugh/dry laugh_2019.06.13_17.20.36.csv'
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
for y in range(0,5):
    mbohtalah = mbohtalah+(5*128)
    ranges.append(mbohtalah)
    mbohtalah = mbohtalah+(1*128)
    ranges.append(mbohtalah)
    mbohtalah = mbohtalah+(5*128)
    ranges.append(mbohtalah)
    mbohtalah = mbohtalah+(3.15*128)
    ranges.append(mbohtalah)
# for y in range(0,5):
#     mbohtalah = mbohtalah+(5*128)
#     ranges.append(mbohtalah)
#     mbohtalah = mbohtalah+(2*128)
#     ranges.append(mbohtalah)
#     mbohtalah = mbohtalah+(5*128)
#     ranges.append(mbohtalah)
#     mbohtalah = mbohtalah+(2.15*128)
#     ranges.append(mbohtalah)
ds1 = []
ds2 = []
for z in range(0,len(ranges)-1):
    if z%4 == 0:
        ds1.append(data[int(round(ranges[z])):int(round(ranges[z+1]))])
    elif z%4 == 2:
        ds2.append(data[int(round(ranges[z])):int(round(ranges[z+1]))])
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
for ww in range(0,5):
    data = np.array(ds1)[ww,:,4]
    # data = data * np.hamming(640)
    time = np.arange(data.size) / sf
    plt.plot(time, data, lw=1.5, color='k', alpha=0.5)
# data = np.mean(ds1,axis=0)[:,0]
# data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
time = np.arange(data.size) / sf
plt.plot(time, data, lw=1.5, color='k', alpha=0.5)
for ww in range(0,5):
    data = np.array(ds2)[ww,:,4]
    # data = data * np.hamming(640)
    time = np.arange(data.size) / sf
    plt.plot(time, data, lw=1.5, color='b', alpha=0.5)
# data = np.mean(ds2,axis=0)[:,0]
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
for ww in range(0,5):
    data = np.array(ds1)[ww,:,4]
    freqs, psd = signal.welch(data, sf, nperseg=win)
    # psd = psd*psd
    plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
# data = np.mean(ds1,axis=0)[:,0]
# data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
freqs, psd = signal.welch(data, sf, nperseg=win)
plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
for ww in range(0,5):
    data = np.array(ds2)[ww,:,4]
    freqs, psd = signal.welch(data, sf, nperseg=win)
    # psd = psd*psd
    plt.plot(freqs, psd, color='b', lw=1, alpha=0.5)
# data = np.mean(ds2,axis=0)[:,0]
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
