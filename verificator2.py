import pickle
import numpy as np
type = 'short no laugh 1'
# type = 'short no laugh 2'
# with open('Z:/nani/experiment/FP_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     data = pickle.load(f)
# data = data[0]
# a = data
import matplotlib.pyplot as plt
# t = np.arange(-3,10,1/128)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
# plt.title("False Positive")
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
# print(np.array(data).shape)
# with open('Z:/nani/experiment/FN_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     data = pickle.load(f)
# data = data[0]
# b = data
# import matplotlib.pyplot as plt
# t = np.arange(-3,10,1/128)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
#     # a = data
# plt.title("False Negative")
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
# print(np.array(data).shape)
with open('Z:/nani/experiment/TP_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    data = pickle.load(f)
data = data[0]
c = data
import matplotlib.pyplot as plt
t = np.arange(-3,10,1/128)
lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
plt.plot(t,lolz[:,:])

    # b = data
plt.title("True Positive")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.ylim(4100,4225)
plt.show()
print(np.array(data).shape)
with open('Z:/nani/experiment/TN_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    data = pickle.load(f)
data = data[0]
d = data
t = np.arange(-3,10,1/128)
lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
plt.plot(t,lolz[:,:])

plt.title("True Negative")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.ylim(4100,4225)
plt.show()
print(np.array(data).shape)

data = np.concatenate((b,c),axis=0)
lolz = np.mean(data,axis=0)
plt.plot(t,lolz[:,[4,9]])
plt.title("Pressed as Funny")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.show()
print(np.array(data).shape)
data = np.concatenate((a,d),axis=0)
lolz = np.mean(data,axis=0)
plt.plot(t,lolz[:,[4,9]])
plt.title("Pressed as Not Funny")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.show()
print(np.array(data).shape)
# data = np.concatenate((a[0], b[0]), axis=0)
# data = data[0]
# import matplotlib.pyplot as plt
# t = np.arange(-3,10,1/128)
# print(np.array(data).shape)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
exit()
truthtable1 =[["TN","TP","TN","FN","TN","TP","FP","TP","TN","TP"],
["FP","TP","FP","TP","TN","TP","FP","TP","FP","TP"],
["TN","TP","TN","FN","TN","TP","FP","FN","FP","TP"],
["TN","TP","TN","FN","TN","TP","FP","FN","FP","FN"],
["TN","TP","TN","FN","TN","TP","TN","TP","TN","TP"],
["TN","FN","FP","TP","TN","TP","FP","TP","TN","TP"],
["FP","TP","FP","FN","TN","FN","TN","TP","TN","TP"],
["TN","TP","FP","TP","TN","TP","TN","FN","TN","TP"],
["TN","FN","TN","TP","TN","TP","FP","FN","FP","TP"],
["TN","TP","FP","FN","TN","FN","TN","FN","TN","FN"],
["TN","FN","TN","FN","TN","FN","TN","FN","TN","TP"],
["TN","TP","FP","FN","TN","TP","FP","FN","TN","TP"],
["TN","FN","FP","TP","TN","TP","TN","FN","TN","TP"],
["TN","FN","TN","FN","TN","FN","TN","FN","TN","TP"],
["TN","FN","TN","FN","TN","TP","TN","FN","TN","FN"],
["TN","FN","FP","FN","TN","TP","TN","FN","FP","FN"],
["TN","TP","TN","TP","TN","TP","TN","FN","TN","TP"],
["TN","FN","FP","TP","TN","TP","FP","TP","FP","TP"],
["TN","FN","FP","TP","TN","FN","FP","TP","TN","TP"]]


truthtable2 =[["TN","TP","FP","TN","TN","TN","TP","FP","TP","TN"],
["TN","FN","TN","TN","TN","TN","TP","TN","FN","TN"],
["TN","TP","TN","TN","TN","TN","TP","TN","TP","TN"],
["TN","TP","TN","TN","TN","TN","TP","TN","TP","TN"],
["TN","TP","TN","TN","TN","TN","TP","TN","TP","FP"],
["TN","TP","TN","TN","FP","FP","TP","TN","TP","TN"],
["TN","TP","TN","TN","TN","TN","TP","TN","TP","TN"],
["TN","FN","TN","TN","TN","FP","TP","TN","TP","TN"],
["TN","FN","TN","TN","TN","TN","TP","TN","TP","FP"],
["TN","FN","TN","TN","TN","TN","TP","TN","FN","TN"],
["TN","FN","TN","TN","TN","TN","TP","TN","TP","TN"],
["TN","TP","TN","TN","FP","TN","FN","TN","TP","TN"],
["TN","FN","TN","TN","TN","TN","TP","TN","FN","TN"],
["TN","FN","TN","TN","FP","TN","FN","TN","FN","TN"],
["TN","FN","TN","TN","TN","TN","FN","TN","FN","TN"],
["TN","FN","TN","TN","TN","TN","FN","TN","TP","TN"],
["TN","FN","TN","TN","TN","TN","FN","TN","FN","TN"],
["TN","FN","FP","FP","TN","FP","TP","TN","TP","TN"],
["TN","TP","TN","TN","TN","TN","FN","TN","FN","TN"],
["TN","TP","TN","TN","TN","TN","FN","TN","TP","TN"],
["TN","FN","TN","TN","FP","TN","TP","TN","TP","TN"]]


sn = [
'gav',
'rot',
'ovi',
'ila',
'hrz',
'aldoh',
'cra',
'cips',
'vyn',
'rijuu',
'sinlo',
'skot',
'manai',
'nature',
'fira',
'kirk',
'prg',
'alin',
'cdef',
'ovi2',
'key'
]
TN = []
FN = []
TP = []
FP = []
# type = 'short no laugh 1'
type = 'short no laugh 2'
for x in range(0,len(sn)):
    file = 'Z:/nani/experiment/'+sn[x]+'/'+type+'.pkl'
    # file = 'Z:/nani/experiment/sinlo/short laugh 1.pkl'
    data = []
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        data = pickle.load(f)
    data = np.array(data)
    data = data[0]
    print(data.shape)
    # exit()
    for y in range(0,10):
        # if truthtable1[x][y]=="TN":
        #     TN.append(data[y,:,:])
        # elif truthtable1[x][y]=="FN":
        #     FN.append(data[y,:,:])
        # elif truthtable1[x][y]=="TP":
        #     TP.append(data[y,:,:])
        # elif truthtable1[x][y]=="FP":
        #     FP.append(data[y,:,:])
        if truthtable2[x][y]=="TN":
            TN.append(data[y,:,:])
        elif truthtable2[x][y]=="FN":
            FN.append(data[y,:,:])
        elif truthtable2[x][y]=="TP":
            TP.append(data[y,:,:])
        elif truthtable2[x][y]=="FP":
            FP.append(data[y,:,:])
with open('Z:/nani/experiment/TN_'+type+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([TN],f)
with open('Z:/nani/experiment/FN_'+type+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([FN],f)
with open('Z:/nani/experiment/TP_'+type+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([TP],f)
with open('Z:/nani/experiment/FP_'+type+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([FP],f)
print("done")
