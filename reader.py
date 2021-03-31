import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import seaborn as sns
import pywt
from sklearn.decomposition import FastICA
import pickle
datayes = []
datano = []
korban = ['aldoh','cips','cra','gav','hrz','ila','ovi','rot']
for e in range(0, len(korban)):
    data = genfromtxt('Z:/nani/experiment/'+korban[e]+'/dry laugh/yes.csv', delimiter=',')
    data2 = genfromtxt('Z:/nani/experiment/'+korban[e]+'/dry laugh/no.csv', delimiter=',')
    data = data[:,2:16]
    data = data.reshape((5,640,14))
    data2 = data2[:,2:16]
    data2 = data2.reshape((5,640,14))
    datayes.append(data)
    datano.append(data2)
with open('Z:/nani/experiment/arr_dry.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([datayes, datano],f)
exit()
"""
ica = FastICA(n_components=14, max_iter=500)
X1 = ica.fit_transform(data[4,:,:])

t = np.linspace(0, 5, 200, endpoint=False)
#sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 50)
sig = X1
cwtmatr = signal.cwt(sig, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[0, 5, 50, 1], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

data = np.mean(data[3:],axis=0)
"""
for i in range(0,5):
    ica = FastICA(n_components=14, max_iter=500)
    data[i] = ica.fit_transform(data[i])

print(data.shape)


data = np.mean(data,axis=0)

coef, freqs=pywt.cwt(data[:,1],np.arange(1,50),'morl')
plt.matshow(coef) # doctest: +SKIP
plt.show() # doctest: +SKIP

# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# plt.plot(data)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Voltage')
# plt.title('N3 sleep EEG data (F3)')
# sns.despine()
# plt.show()



exit()

if len(dataavg)==0:
    dataavg = data
else:
    dataavg += data
data2 = genfromtxt('Z:/nani/experiment/cips/dry laugh/no.csv', delimiter=',')
data2 = data2[0,128:1152]
if len(dataavg2)==0:
    dataavg2 = data2
else:
    dataavg2 += data2
