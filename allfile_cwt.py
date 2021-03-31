import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
import pywt
# data = genfromtxt('Z:/nani/experiment/aldoh/dry laugh/dry laugh_2019.06.01_12.26.08.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/aldoh/funny laugh/funny laugh_2019.06.01_12.33.38.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/aldoh/short laugh 2/short laugh 2_2019.06.01_11.56.53.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/cra/funny laugh/funny laugh_2019.06.02_14.22.42.csv', skip_header=1, delimiter=',')
# data = genfromtxt('Z:/nani/experiment/ila/dry laugh/dry laugh_2019.05.22_17.47.10.csv', skip_header=1, delimiter=',')
# loc = 'Z:/nani/experiment/ila/dry laugh/dry laugh_2019.05.22_17.47.10.csv'
# loc = 'Z:/nani/experiment/sinlo/dry laugh/dry laugh_2019.06.10_15.12.31.csv'
# loc = 'Z:/nani/experiment/sinlo/funny laugh/funny laugh_2019.06.10_15.15.58.csv'
loc = 'Z:/nani/experiment/sinlo/short laugh 1/short laugh 1_2019.06.10_14.56.21.csv'
# loc = 'Z:/nani/experiment/ovi/funny laugh/forced laugh funny_201  9.05.22_12.32.52.csv'
# loc = 'Z:/nani/experiment/ila/funny laugh/forced funny laugh_2019.05.22_17.55.35.csv'
# loc = 'Z:/nani/experiment/aldoh/funny laugh/funny laugh_2019.06.01_12.33.38.csv'
# loc = 'Z:/nani/experiment/cra/funny laugh/funny laugh_2019.06.02_14.22.42.csv'
# loc = 'Z:/nani/experiment/cra/dry laugh/dry laugh_2019.06.02_14.18.15.csv'
data = genfromtxt(loc, skip_header=1, delimiter=',')
df = pd.read_csv(loc, header=None, skiprows=1)
df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
peteek = (df.query('t == "100"').index)
startpoint = (df.query('t == "100"').index[0])
data = data[:,2:16]
print(data.shape)
useica = 1
multivalue = 200
if useica==1:
    ica = FastICA(n_components=14, max_iter=500)
    data = ica.fit_transform(data)
    multivalue /= 10000
data = np.transpose(data)
coef, freqs=pywt.cwt(data[1,peteek[0]:peteek[0]+640],np.arange(1,50),'morl')
plt.plot(data[1,peteek[0]:peteek[0]+640])
plt.matshow(coef) # doctest: +SKIP
# plt.axvspan(peteek[0], peteek[1], facecolor='y', alpha=0.5,zorder=1)
plt.show() # doctest: +SKIP
exit()

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
    mbohtalah = mbohtalah+(3.2*128)
    ranges.append(mbohtalah)
for mbuoh in range(0,20):
    if mbuoh%4 ==0:
        plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='g', alpha=0.5,zorder=1)
    elif mbuoh%4 ==1:
        plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='b', alpha=0.5,zorder=1)
    elif mbuoh%4 ==2:
        plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='r', alpha=0.5,zorder=1)
    else:
        plt.axvspan(ranges[mbuoh], ranges[mbuoh+1], facecolor='b', alpha=0.5,zorder=1)
plt.axvspan(peteek[0], peteek[1], facecolor='y', alpha=0.5,zorder=1)
plt.plot(data)
plt.show()
