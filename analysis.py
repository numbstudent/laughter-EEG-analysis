import numpy as np
import pickle
from matplotlib import pyplot as plt
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FastICA

# with open('Z:/nani/experiment/arr_dry.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
with open('Z:/nani/experiment/arr_dry_ica.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
# with open('Z:/nani/experiment/arr_funny.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
# with open('Z:/nani/experiment/arr_funny_ica.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    ds1, ds2 = pickle.load(f)
ds1 = np.array(ds1).reshape((40,640,14))
ds2 = np.array(ds2).reshape((40,640,14))

data = np.mean(ds2,axis=0)
# for i in range(0,40):
#     data = ds1[i]
#     plt.plot(data)
#     plt.pause(1)
#     plt.clf()
plt.specgram(data[:,4],Fs=128)
#import pywt
# coef, freqs=pywt.cwt(data[:,4],np.arange(1,50),'morl')
#coef, freqs=pywt.dwt(data[:,9],'haar')
#plt.plot(data[:,9]  ) # doctest: +SKIP
#plt.show() # doctest: +SKIP
# plt.matshow(coef) # doctest: +SKIP
#plt.plot(coef) # doctest: +SKIP
plt.show() # doctest: +SKIP
exit()
ch = 9
ds1 = ds1[:,:,ch]
ds2 = ds2[:,:,ch]

print(np.array(ds1).shape)
print(np.array(ds2).shape)

labels = []
x_dump = []
#1 yes 0 no
for i in range(0,40):
    labels.append(0)
    labels.append(1)
    x_dump.append(ds2[i])
    x_dump.append(ds1[i])



X = np.array(x_dump)
y = np.array(labels)

print(X.shape)
print(y.shape)
from sklearn.model_selection import cross_val_score
clf = SVC(kernel='rbf')
scores = cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
exit()
