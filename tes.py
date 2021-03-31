import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn.decomposition import FastICA

data = genfromtxt('Z:/nani/experiment/gav/dry laugh/yes.csv', delimiter=',')
data = data[:,2:16]
# data = np.transpose(data)
data = data-4100
ica = FastICA(n_components=14, max_iter=1500, tol = 0.5, random_state=0)
data = ica.fit_transform(data)
plt.plot(data[:,9],'black',alpha=0.5)
data = genfromtxt('Z:/nani/experiment/gav/dry laugh/no.csv', delimiter=',')
data = data[:,2:16]
# data = np.transpose(data)
data = data-4100
ica = FastICA(n_components=14, max_iter=1500, tol = 0.5, random_state=0)
data = ica.fit_transform(data)
plt.plot(data[:,9],'blue',alpha=0.5)
# plt.show()
data = genfromtxt('C:/Users/freesugar/Downloads/EEG-Classification-master/data/ML105_US.csv', delimiter=',')
data = np.transpose(data)
# plt.plot(data[:640,:])
print(data.shape)
plt.show()
