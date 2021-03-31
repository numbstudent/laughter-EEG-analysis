import pickle
import numpy as np
type = 'short laugh 1'
# type = 'short laugh 2'
# with open('Z:/nani/experiment/TP_short no laugh 2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
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
# with open('Z:/nani/experiment/TN_short no laugh 2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
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
with open('Z:/nani/experiment/TP_short laugh 1.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    data = pickle.load(f)
data = data[0]
c = data
import matplotlib.pyplot as plt
t = np.arange(-3,10,1/128)
lolz = np.mean(data,axis=0)
plt.plot(t,lolz[:,[4,9]])
    # b = data
plt.title("True Positive")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.ylim(4130,4170)

plt.show()
print(np.array(data).shape)
with open('Z:/nani/experiment/TN_short laugh 1.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    data = pickle.load(f)
data = data[0]
d = data
t = np.arange(-3,10,1/128)
lolz = np.mean(data,axis=0)
plt.plot(t,lolz[:,[4,9]])
plt.title("True Negative")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.ylim(4130,4170)

plt.show()
print(np.array(data).shape)
data = np.concatenate((a,c),axis=0)
lolz = np.mean(data,axis=0)
plt.plot(t,lolz[:,[4,9]])
plt.title("Without Canned Laugh Funny")
plt.xlabel("Time")
plt.ylabel("Raw Value")

plt.show()
print(np.array(data).shape)
data = np.concatenate((b,d),axis=0)
lolz = np.mean(data,axis=0)
plt.plot(t,lolz[:,[4,9]])
plt.title("Without Canned Laugh Not Funny")
plt.xlabel("Time")
plt.ylabel("Raw Value")
plt.ylim(4130,4180)
plt.show()
print(np.array(data).shape)
exit()
# # type = 'short laugh 1'
# type = 'short laugh 2'
# with open('Z:/nani/experiment/FP_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     data = pickle.load(f)
# data = data[0]
# a = data
# import matplotlib.pyplot as plt
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
# with open('Z:/nani/experiment/TP_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     data = pickle.load(f)
# data = data[0]
# c = data
# import matplotlib.pyplot as plt
# t = np.arange(-3,10,1/128)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
#     # b = data
# plt.title("True Positive")
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
# print(np.array(data).shape)
# with open('Z:/nani/experiment/TN_'+type+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     data = pickle.load(f)
# data = data[0]
# d = data
# t = np.arange(-3,10,1/128)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
# plt.title("True Negative")
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
# print(np.array(data).shape)
# data = np.concatenate((b,c),axis=0)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
# plt.title("Pressed as Funny")
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
# print(np.array(data).shape)
# data = np.concatenate((a,d),axis=0)
# lolz = np.mean(data,axis=0)
# plt.plot(t,lolz[:,[4,9]])
# plt.title("Pressed as Not Funny")
# plt.xlabel("Time")
# plt.ylabel("Raw Value")
# plt.show()
# print(np.array(data).shape)
# # data = np.concatenate((a[0], b[0]), axis=0)
# # data = data[0]
# # import matplotlib.pyplot as plt
# # t = np.arange(-3,10,1/128)
# # print(np.array(data).shape)
# # lolz = np.mean(data,axis=0)
# # plt.plot(t,lolz[:,[4,9]])
# # plt.xlabel("Time")
# # plt.ylabel("Raw Value")
# # plt.show()
# exit()
truthtable1 =[["TP","TN","FP","TP","TN","TN","TP","FN","FN","FP"],
["TP","TN","FP","TP","TN","FP","FN","TP","TP","TN"],
["FN","TN","FP","TP","TN","TN","TP","TP","TP","FP"],
["TP","FP","FP","FN","TN","TN","FN","TP","TP","TN"],
["TP","TN","FP","TP","FP","TN","TP","TP","TP","TN"],
["FN","TN","FP","TP","TN","TN","TP","TP","TP","TN"],
["FN","TN","FP","TP","FP","TN","TP","TP","FN","TN"],
["TP","FP","TN","FN","FP","FP","TP","TP","FN","FP"],
["TP","FP","FP","FN","TN","TN","FN","FN","FN","TN"],
["FN","TN","TN","TP","TN","TN","TP","FN","TP","TN"],
["TP","FP","TN","FN","TN","TN","FN","FN","FN","TN"],
["FN","TN","TN","FN","TN","TN","FN","FN","FN","TN"],
["FN","TN","TN","TP","FP","TN","TP","TP","FN","TN"],
["FN","TN","TN","FN","TN","TN","FN","FN","TP","TN"],
["TP","TN","TN","TP","TN","TN","FN","FN","FN","TN"],
["TP","TN","TN","FN","TN","TN","FN","FN","FN","FP"],
["TP","FP","TN","TP","TN","TN","TP","TP","FN","FP"],
["FN","TN","TN","FN","TN","TN","FN","FN","FN","FP"],
["FN","FP","FP","TP","TN","TN","TP","TP","TP","TN"],
["FN","FP","FP","FN","FP","TN","TP","TP","TP","TN"]]


truthtable2 =[
# ["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],
["FP","TN","TP","TN","TP","TP","FN","TN","FP","TP"],
["TN","TN","TP","TN","TP","TP","TP","FP","TN","TP"],
["TN","TN","TP","TN","TP","TP","TP","TN","TN","TP"],
["TN","TN","TP","FP","TP","TP","TP","TN","TN","FN"],
["TN","TN","TP","TN","TP","TP","FN","FP","TN","TP"],
["TN","TN","FN","TN","TP","TP","TP","TN","FP","TP"],
["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],
["TN","TN","TP","TN","TP","TP","TP","FP","FP","TP"],
# ["FP","TN","TP","TN","FN","TP","TP","TN","TN","FN"],
["TN","TN","TP","TN","TP","FN","TP","TN","TN","TP"],
["TN","TN","TP","FP","TP","TP","TP","FP","FP","TP"],
["TN","TN","FN","TN","FN","FN","FN","TN","TN","FN"],
["TN","TN","TP","TN","TP","TP","FN","TN","TN","TP"],
["TN","TN","TP","TN","FN","TP","FN","TN","TN","FN"],
["TN","TN","TP","TN","FN","FN","TP","TN","TN","TP"],
["TN","TN","FN","TN","FN","FN","FN","TN","TN","FN"],
["TN","FP","TP","TN","TP","TP","TP","FP","TN","TP"],
["TN","TN","FN","TN","TP","TP","FN","TN","TN","FN"],
["TN","TN","TP","TN","TP","TP","TP","FP","TN","TP"],
["TN","FP","FN","TN","TP","TP","TP","FP","FP","FN"]]#ovi2

# [["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],
# ["FP","TN","TP","TN","TP","TP","FN","TN","FP","TP"],
# ["TN","TN","TP","TN","TP","TP","TP","FP","TN","TP"],
# ["TN","TN","TP","TN","TP","TP","TP","TN","TN","TP"],
# ["TN","TN","TP","FP","TP","TP","TP","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","TP","FN","FP","TN","TP"],
# ["TN","TN","FN","TN","TP","TP","TP","TN","FP","TP"],
# ["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],
# ["TN","TN","TP","TN","TP","TP","TP","FP","FP","TP"],
# # ["FP","TN","TP","TN","FN","TP","TP","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","FN","TP","TN","TN","TP"],
# ["TN","TN","TP","FP","TP","TP","TP","FP","FP","TP"],
# ["TN","TN","FN","TN","FN","FN","FN","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","TP","FN","TN","TN","TP"],
# ["TN","TN","TP","TN","FN","TP","FN","TN","TN","FN"],
# ["TN","TN","TP","TN","FN","FN","TP","TN","TN","TP"],
# ["TN","TN","FN","TN","FN","FN","FN","TN","TN","FN"],
# ["TN","FP","TP","TN","TP","TP","TP","FP","TN","TP"],
# ["TN","TN","FN","TN","TP","TP","FN","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","TP","TP","FP","TN","TP"],
# ["TN","FP","FN","TN","TP","TP","TP","FP","FP","FN"]]


# [["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],
# ["FP","TN","TP","TN","TP","TP","FN","TN","FP","TP"],
# ["TN","TN","TP","TN","TP","TP","TP","FP","TN","TP"],
# ["TN","TN","TP","TN","TP","TP","TP","TN","TN","TP"],
# ["TN","TN","TP","FP","TP","TP","TP","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","TP","FN","FP","TN","TP"],
# ["TN","TN","FN","TN","TP","TP","TP","TN","FP","TP"],
# ["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],
# ["TN","TN","TP","TN","TP","TP","TP","FP","FP","TP"],
# ["FP","TN","TP","TN","FN","TP","TP","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","FN","TP","TN","TN","TP"],
# ["TN","TN","TP","FP","TP","TP","TP","FP","FP","TP"],
# ["TN","TN","FN","TN","FN","FN","FN","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","TP","FN","TN","TN","TP"],
# ["TN","TN","TP","TN","FN","TP","FN","TN","TN","FN"],
# ["TN","TN","TP","TN","FN","FN","TP","TN","TN","TP"],
# ["TN","TN","FN","TN","FN","FN","FN","TN","TN","FN"],
# ["TN","FP","TP","TN","TP","TP","TP","FP","TN","TP"],
# ["TN","TN","FN","TN","TP","TP","FN","TN","TN","FN"],
# ["TN","TN","TP","TN","TP","TP","TP","FP","TN","TP"],
# ["TN","FP","FN","TN","TP","TP","TP","FP","FP","FN"]]
#key
# [["TN","FN","TP","TN","TP","TP","TP","TN","TN","TP"],
#                 ["FN","TN","TP","TN","TP","TP","FP","TN","FN","TP"],
#                 ["TN","TN","TP","TN","TP","TP","TP","FN","TN","TP"],
#                 ["TN","TN","TP","TN","TP","TP","TP","TN","TN","TP"],
#                 ["TN","TN","TP","FN","TP","TP","TP","TN","TN","FP"],
#                 ["TN","TN","TP","TN","TP","TP","FP","FN","TN","TP"],
#                 ["TN","TN","FP","TN","TP","TP","TP","TN","FN","TP"],
#                 ["TN","FN","TP","TN","TP","TP","TP","TN","TN","TP"],
#                 ["TN","TN","TP","TN","TP","TP","TP","FN","FN","TP"],
#                 ["FN","TN","TP","TN","FP","TP","TP","TN","TN","FP"],
#                 ["TN","TN","TP","TN","TP","FP","TP","TN","TN","TP"],
#                 ["TN","TN","TP","FN","TP","TP","TP","FN","FN","TP"],
#                 ["TN","TN","FP","TN","FP","FP","FP","TN","TN","FP"],
#                 ["TN","TN","TP","TN","TP","TP","FP","TN","TN","TP"],
#                 ["TN","TN","TP","TN","FP","TP","FP","TN","TN","FP"],
#                 ["TN","TN","TP","TN","FP","FP","TP","TN","TN","TP"],
#                 ["TN","TN","FP","TN","FP","FP","FP","TN","TN","FP"],
#                 ["TN","FN","TP","TN","TP","TP","TP","FN","TN","TP"],
#                 ["TN","TN","FP","TN","TP","TP","FP","TN","TN","FP"],
#                 ["TN","TN","TP","TN","TP","TP","TP","FN","TN","TP"],
#                 ["TN","FN","FP","TN","TP","TP","TP","FN","FN","FP"]]
#
#                 ["TN","FP","TP","TN","TP","TP","TP","TN","TN","TP"],


anstable1 = [40,20,20,40,20,20,40,40,40,20]
anstable2 = [20,20,40,20,40,40,40,20,20,20]
sn = [
# 'gav',
'rot',
'ovi',
'ila',
'hrz',
'aldoh',
'cra',
'cips',
'vyn',
# 'rijuu',
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
type = 'short laugh 2'
# type = 'short laugh 2'
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
