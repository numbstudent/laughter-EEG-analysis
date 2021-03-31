import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import simps
name = 'skot'
filename=['gav','ovi','ila','hrz','aldoh','cra','cips','vyn','rijuu','sinlo','skot','manai','nature']
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
if useica==1:
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=14, max_iter=1000)
    for xx in range(0,5):
        X1[xx] = ica.fit_transform(X1[xx])
        X2[xx] = ica.fit_transform(X2[xx])
##################SVM STARTS HERE#########################################
# X = np.concatenate((X1, X2), axis=0)
# y1 = np.ones((len(X1),), dtype=int)
# y2 = np.zeros((len(X2),), dtype=int)
# y = np.concatenate((y1, y2), axis=0)
# print(X.shape)
# from sklearn.model_selection import train_test_split
# basearray = np.arange(len(y))*8
# from random import *
# randomnumber = randint(0, 389)
# counter = basearray[randomnumber]
# x_test = X[counter:counter+8]
# y_test = y[counter:counter+8]
# X = np.delete(X,np.s_[counter:counter+8],axis=0)
# y = np.delete(y,np.s_[counter:counter+8],axis=0)
# print(x_test,y_test)
# # X, x_test, y, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)
# # X, yyyy, y, www = train_test_split(X, y, test_size=0.1)
# from sklearn.svm import SVC
# print(X.shape)
# # print(X.shape)
# # plt.show()
# # plt.scatter(X1[0:1960,3,20:30],   cmap='autumn');
# # exit()
# clf = SVC(kernel='rbf')
# clf.fit(X, y)
# # print(clf.predict([[9, -1]]))
# print(clf.score(x_test, y_test))
# exit()
#############################SVM with cross val#################################
from sklearn.svm import SVC
# X = np.concatenate((X1[0:1960,4,[15]], X2[0:1960,4,[15]]), axis=0)
# X = np.concatenate((X1[:,:,10], X2[:,:,10]), axis=0)
X = np.concatenate((X1[:,:,4], X2[:,:,4]), axis=0)
# X = np.concatenate((np.mean(X1, axis=0), np.mean(X2, axis=0)), axis=0)
# y1 = np.ones((1960,), dtype=int)
# y2 = np.zeros((1960,), dtype=int)
y1 = np.ones((len(X1),), dtype=int)
y2 = np.zeros((len(X2),), dtype=int)
y = np.concatenate((y1, y2), axis=0)
print(X.shape)
from sklearn.model_selection import cross_val_score
clf = SVC(kernel='rbf')
scores = cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
exit()
#############################decision tree with cross val#################################
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# X = np.concatenate((X1[0:1960,4,[14]], X2[0:1960,4,[14]]), axis=0)
# y1 = np.ones((1960,), dtype=int)
# y2 = np.zeros((1960,), dtype=int)
# y = np.concatenate((y1, y2), axis=0)
# print(X.shape)
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(clf, X, y, cv=24)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# exit()
#############################LDA with cross val#################################
# # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# # clf = LinearDiscriminantAnalysis()
# from sklearn.svm import SVC
# clf = SVC(kernel='rbf')
# # from sklearn import tree
# # clf = tree.DecisionTreeClassifier()
# # from sklearn.linear_model import SGDClassifier
# # clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
# # from sklearn.neural_network import MLPClassifier
# # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(5, 2), random_state=1)
# print(np.array(X1).shape)
# X = np.concatenate((X1, X2), axis=0)
# print(np.mean(X1))
# print(np.mean(X2))
# X = np.mean(X, axis=2)
# # X = X*X
# y1 = np.ones((len(X1),), dtype=int)
# y2 = np.zeros((len(X2),), dtype=int)
# y = np.concatenate((y1, y2), axis=0)
# print(X.shape)
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(clf, X, y, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# exit()


# X1 = X1[0:1960,[4],13:17]
# X2 = X2[0:1960,[4],13:17]


# Xtest = X2[1960:,1:4,:]
# X3 = X3[0:245,[4,37,39,12,47,49,20,29,57],:]
# X4 = X4[0:245,[4,37,39,12,47,49,20,29,57],:]
# X1 = X1[:,[1,3],14:30]
# X1 = X1[:,:,30:60]
# X2 = X2[:,[1,3],14:30]
# X2 = X2[:,:,30:60]
# X1 = np.swapaxes(X1,0,1)
# X2 = np.swapaxes(X2,0,1)
# X1 = X1[:245,512:768,0:10] #for alpha, beta, delta, theta, lowgamma
# X2 = X2[:245,512:768,54:64] #for alpha, beta, delta, theta, lowgamma
# # filter = np.concatenate((np.hanning(128),np.hanning(128)))
# filter = np.hamming(256)
# for i in range(0,X1.shape[0]):
#     for j in range(0,X1.shape[2]):
#         X1[i,:,j] = X1[i,:,j]*filter
#         X2[i,:,j] = X2[i,:,j]*filter
#         # plt.plot(X1[i,:,j])
#         # plt.show()
# exit()
print(X1.shape)
print(X2.shape)
# exit()
# X1 = np.swapaxes(X1,1,2)
# X2 = np.swapaxes(X2,1,2)

# plt.plot(X1)
# plt.show()x`
# plt.plot(X2)
# plt.show()
# exit()
X = np.concatenate((X1, X2), axis=0)
# X3 = np.swapaxes(X3,1,2)
# X4 = np.swapaxes(X4,1,2)
# Xy = np.concatenate((X3, X4), axis=0)

# y1 = np.ones((245,), dtype=int)
# y2 = np.zeros((245,), dtype=int)
# y1 = np.ones((1960,), dtype=int)
# y2 = np.zeros((1960,), dtype=int)
y1 = np.ones((len(X1),), dtype=int)
y2 = np.zeros((len(X2),), dtype=int)
y = np.concatenate((y1, y2), axis=0)

# X = np.concatenate((X, X[:50]), axis=0)
# y = np.concatenate((y, y[:50]), axis=0)
# X = np.concatenate((X, X[:50]), axis=0)
# y = np.concatenate((y, y[:50]), axis=0)
# X = np.concatenate((X, X[:50]), axis=0)
# y = np.concatenate((y, y[:50]), axis=0)
# X = np.concatenate((X, X[:50]), axis=0)
# y = np.concatenate((y, y[:50]), axis=0)
# X = np.concatenate((X, X[:25]), axis=0)
# y = np.concatenate((y, y[:25]), axis=0)
# X = np.concatenate((X, X[:25]), axis=0)
# y = np.concatenate((y, y[:25]), axis=0)
# X = np.concatenate((X, X[:25]), axis=0)
# y = np.concatenate((y, y[:25]), axis=0)
# X = np.concatenate((X, X[:25]), axis=0)
# y = np.concatenate((y, y[:25]), axis=0)
# X = np.concatenate((X, X[:25]), axis=0)
# y = np.concatenate((y, y[:25]), axis=0)

print(X.shape)
# Downsample, shuffle and split (from sklearn.cross_validation)
from sklearn.model_selection import train_test_split
X, x_test, y, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
# X, x_test, y, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
# Xy, x_testy, X, x_test, y, y_test = train_test_split(Xy, X, y, test_size=0.25)
# x_trainy, x_valy, x_train, x_val, y_train, y_val = train_test_split(Xy, X, y, test_size=0.25)
#
# print(y_train)
# print(y_val)
# print(y_test)
# exit()
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling1D, Input, concatenate
from keras.layers import Conv1D, MaxPooling1D, SpatialDropout1D, CuDNNLSTM, RepeatVector, TimeDistributed, LSTM
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import optimizers

num_classes = 2

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)


# Create the network
model = keras.models.Sequential()
model.add(keras.layers.normalization.BatchNormalization(input_shape=(X.shape[1],X.shape[2])))
model.add(keras.layers.core.Dense(64, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.2))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(64, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.core.Flatten())
# opt = optimizers.rmsprop(lr=0.00001)
model.add(keras.layers.core.Dense(2,   activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='rmsprop',metrics=["accuracy"])
print(model.summary())

# model = keras.models.Sequential()
# model.add(keras.layers.normalization.BatchNormalization(input_shape=(X.shape[1],X.shape[2])))
# model.add(keras.layers.core.Dense(16, activation='relu'))
# model.add(keras.layers.core.Dropout(rate=0.5))
# model.add(keras.layers.normalization.BatchNormalization())
# model.add(keras.layers.core.Dense(16, activation='relu'))
# model.add(keras.layers.core.Dropout(rate=0.5))
# model.add(keras.layers.normalization.BatchNormalization())
# model.add(keras.layers.core.Dense(16, activation='relu'))
# model.add(keras.layers.core.Dropout(rate=0.5))
# model.add(keras.layers.core.Flatten())
# model.add(keras.layers.core.Dense(2,   activation='sigmoid'))
# model.compile(loss="categorical_crossentropy", optimizer="adadelta",metrics=["accuracy"])
# print(model.summary())


# model = Sequential()
# model.add(Conv1D(64, 3, activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(MaxPooling1D())
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(256, 3, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model = Sequential()
# model.add(Conv1D(64, 3, activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(MaxPooling1D(2))
# model.add(SpatialDropout1D(rate=0.01))
# model.add(Conv1D(128, 3, activation='relu'))
# # model.add(Conv1D(128, 3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(SpatialDropout1D(rate=0.01))
# # model.add(Conv1D(256, 3, activation='relu'))
# # model.add(Conv1D(256, 3, activation='relu'))
# # model.add(MaxPooling1D(2))
# # model.add(SpatialDropout1D(rate=0.01))
# model.add(Flatten())
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2, activation='sigmoid'))
# # opt = optimizers.Adam(lr=0.001)
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# model = Sequential()
# model.add(Conv1D(16, 5, padding='valid', activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(Conv1D(16, 5, padding='valid', activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(SpatialDropout1D(rate=0.01))
# model.add(Conv1D(32, 3, padding='valid', activation='relu'))
# model.add(Conv1D(32, 3, padding='valid', activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(SpatialDropout1D(rate=0.01))
# model.add(Conv1D(32, 3, padding='valid', activation='relu'))
# model.add(Conv1D(32, 3, padding='valid', activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(SpatialDropout1D(rate=0.01))
# model.add(Conv1D(256, 3, padding='valid', activation='relu'))
# model.add(Conv1D(256, 3, padding='valid', activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2, activation='softmax'))
# opt = optimizers.Adam(lr=0.001)
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# model = Sequential()
# model.add(Conv1D(20, 5, activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(Conv1D(20, 5, activation='relu'))
# model.add(MaxPooling1D(2))
# # model.add(Conv1D(40, 5, activation='relu'))
# # model.add(MaxPooling1D(2))
# # model.add(Conv1D(40, 5, activation='relu'))
# # model.add(MaxPooling1D(2))
# # model.add(Conv1D(160, 10, activation='relu'))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))
# opt = optimizers.Adam(lr=0.0001)
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# model = Sequential()
# model.add(Conv1D(32, 3, activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(Conv1D(32, 3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(32, 3, activation='relu'))
# model.add(MaxPooling1D(2))
# # model.add(Conv1D(32, 3, activation='relu'))
# # model.add(MaxPooling1D(2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))
# opt = optimizers.Adam(lr=0.0001)
# # opt = optimizers.rmsprop(lr=0.001, decay=1e-6)
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(10))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(200, activation='relu', return_sequences=True))
# model.add(Flatten())
# model.add(RepeatVector(2))
# model.add(TimeDistributed(Dense(100, activation='relu')))
# model.add(TimeDistributed(Dense(50, activation='relu')))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# opt = optimizers.Adam(lr=0.001)
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1],X.shape[2])))
# model.add(LSTM(200, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(100, activation='relu')))
# model.add(TimeDistributed(Dense(50, activation='relu')))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# opt = optimizers.Adam(lr=0.001)
# print(model.summary())
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# visible = Input(shape=(X.shape[1],X.shape[2]))
# visible2 = Input(shape=(Xy.shape[1],Xy.shape[2]))
# # first interpretation model
# interp1 = Conv1D(16, 5, padding='valid', activation='relu')(visible)
# interp1 = Dense(10, activation='relu')(interp1)
# interp1 = Flatten()(interp1)
# # second interpretation model
# interp2 = Conv1D(16, 5, padding='valid', activation='relu')(visible2)
# interp2 = Dense(10, activation='relu')(interp2)
# interp2 = Flatten()(interp2)
# # merge interpretation
# merge = concatenate([interp1, interp2])
# # output
# output = Dense(2, activation='sigmoid')(merge)
# model = Model(inputs=[visible, visible2], outputs=output)
# # summarize layers
# print(model.summary())
#
# opt = optimizers.Adam(lr=0.00008)
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# model.add(Conv1D(40, 10, strides=5, padding='same', input_shape=(X.shape[1],X.shape[2])))
# print(model.summary())
# model.add(Activation('relu'))
#
# model.add(Dropout(0.2))
# model.add(MaxPooling1D(3))
# model.add(Conv1D(40, 5, strides=2, padding='same'))
# # model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling1D(3))
# model.add(Conv1D(40, 4, strides=1, padding='same'))
# model.add(Dropout(0.2))
# # model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling1D(3))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
# opt = optimizers.Adam(lr=0.00008)
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])


# Train and save results for later plotting
history = model.fit(x_train, y_train,
    batch_size=100, epochs=300, validation_data=(x_val,y_val))
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
