import numpy as np
import seaborn as sns
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
# loc = 'Z:/nani/experiment/cdef/short laugh 1/short laugh 1_2019.06.25_14.02.31.csv'
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
for ctr in range(0,len(sn)):
    name = sn[ctr]
    categories = ['short laugh 1','short laugh 2','short no laugh 1','short no laugh 2']
    # type = 'short laugh 1'
    # type = 'short laugh 2'
    # type = 'short no laugh 1'
    # type = 'short no laugh 2'
    type = 'funny laugh'
    # type = 'dry laugh'
    # for ctr2 in range(0,len(categories)):
    #     type = categories[ctr2]
    loc = 'Z:/nani/experiment/'+sn[ctr]+'/'+type+'/'+type+'.csv'
    data = genfromtxt(loc, skip_header=1, delimiter=',')
    df = pd.read_csv(loc, header=None, skiprows=1)
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    peteek = (df.query('t == "100"').index)
    print(peteek)
    print(sn[ctr],' --> ')#,peteek[1]-peteek[0])
    # f=open("status.txt","a+")
    # f.write(type+' , '+sn[ctr]+' --> '+str(peteek[1]-peteek[0])+'\n')
    # f.close()
    # exit()
    startpoint = (df.query('t == "100"').index[0])
    # startpoint = startpoint +128 #adding 1 second after
    # startpoint = 0
    data = data[:,2:16]
    print('datashape: ',data.shape)
    useica = 0
    multivalue = 200
    if useica==1:
        ica = FastICA(n_components=14, max_iter=500)
        data = ica.fit_transform(data)
        multivalue /= 10000
    # for x in range(0,14):
    #     data[:,x] = data[:,x] + (multivalue*x)
    mbohtalah = startpoint
    ranges = []
    print("startpoint = ",startpoint)
    if type=='short laugh 1':
        ############################################### 1 short laugh 1
        mbohtalah = startpoint+(1427)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(3880)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(6538)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(8960)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(11422)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(13991)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(16411)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(18784)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(21304)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(23593)
        # mbohtalah = startpoint+(25880)
        ranges.append(mbohtalah)
        # mbohtalah = startpoint+(11.2*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(30.53*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(51.318*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(70.361*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(89.844*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(110.016*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(129.214*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(148.051*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(167.716*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(185.891*128)
        # ranges.append(mbohtalah)
    elif type=='short laugh 2':
        ############################################### 2 short laugh 2
        mbohtalah = startpoint+(1765)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(4361)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(6964)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(9474)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(12285)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(14986)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(17597)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(20175)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(22658)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(25487)
        ranges.append(mbohtalah)
        # mbohtalah = startpoint+(13.851*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(34.159*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(54.750*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(74.452*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(96.628*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(117.873*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(138.421*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(158.781*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(178.254*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(202.707*128)
        # ranges.append(mbohtalah)
    elif type=='short no laugh 1':
        ############################################### 3 fixed short no laugh 1
        mbohtalah = startpoint+(1114)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(3696)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(6352)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(8656)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(11127)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(13605)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(16127)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(18516)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(21409)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(23906)
        ranges.append(mbohtalah)
        # mbohtalah = startpoint+(8.774*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(28.973*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(49.816*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(67.914*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(87.408*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(106.912*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(126.947*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(145.624*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(168.669*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(188.145*128)
        # ranges.append(mbohtalah)
    elif type=='short no laugh 2':
        ############################################## 4 short no laugh 2
        mbohtalah = startpoint+(1281)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(3836)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(6391)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(8905)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(11384)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(13824)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(16510)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(19114)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(21695)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(24478)
        ranges.append(mbohtalah)
        # mbohtalah = startpoint+(10.016*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(29.986*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(50.192*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(69.968*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(89.492*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(108.477*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(129.832*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(150.324*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(170.511*128)
        # ranges.append(mbohtalah)
        # mbohtalah = startpoint+(192.641*128)
        # ranges.append(mbohtalah)
    elif type=='funny laugh' or type=='dry laugh':
        mbohtalah = startpoint+(0)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(797)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(1801)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(2626)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(3609)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(4433)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(5425)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(6239)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(7227)
        ranges.append(mbohtalah)
        mbohtalah = startpoint+(8036)
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
    # f=open("yesnonew.txt","a+")
    # f.write(type+' , '+sn[ctr]+'\n')
    # f.write(str(ranges))
    # f.write('\n')
    # f.write(str(np.array(df.query('t == "20" or t == "40"').index)))
    # f.write('\n')
    # f.write(str(np.array(df.query('t == "20" or t == "40"')['t'])))
    # f.write('\n')

    # print(ranges)
    # print(df.query('t == "20" or t == "40"').index)
    # print(np.array(df.query('t == "20" or t == "40"')['t']))
    # f.close()
    if type=='funny laugh' or type=='dry laugh':
        for z in range(0,len(ranges)):
            if z%2==0:
                print("range= ",int(round(ranges[z]))," - ",int(round(ranges[z]))+(5*128))
                print("stop = ",ranges[z])
                ds1.append(data[int(round(ranges[z])):int(round(ranges[z]))+(5*128)])
            if z%2==1:
                print("range= ",int(round(ranges[z]))," - ",int(round(ranges[z]))+(5*128))
                print("stop = ",ranges[z])
                ds2.append(data[int(round(ranges[z])):int(round(ranges[z]))+(5*128)])
    else:
        for z in range(0,len(ranges)):
            print("range= ",int(round(ranges[z]))-(3*128)," - ",int(round(ranges[z]))+(8*128))
            print("stop = ",ranges[z])
            ds1.append(data[int(round(ranges[z]))-(3*128):int(round(ranges[z]))+(10*128)])
    print(np.array(ds1).shape)
    import pickle
    print(name)
    if np.array(ds1).shape[2]==0:
        print(name)
        exit()
    if np.array(ds2).shape[2]==0:
        print(name)
        exit()
    with open('Z:/nani/experiment/'+name+'/'+type+'_yes.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ds1],f)
    with open('Z:/nani/experiment/'+name+'/'+type+'_no.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ds2],f)
    print('OK')
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
    # import numpy as np
    # # data = np.array(ds1).reshape((3200,14))[:,9]
    # # data2 = np.array(ds2).reshape((3200,14))[:,9]
    # # data = np.array(ds1)[1,:,9]
    # # data2 = np.array(ds1)[2,:,9]
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set(font_scale=1.2)
    #
    # # Define sampling frequency and time vector
    # sf = 128.
    #
    # # Plot the signal
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # for ww in range(0,5):
    #     data = np.array(ds1)[ww,:,4]
    #     # data = data * np.hamming(640)
    #     time = np.arange(data.size) / sf
    #     plt.plot(time, data, lw=1.5, color='k', alpha=0.5)
    # # data = np.mean(ds1,axis=0)[:,0]
    # # data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
    # time = np.arange(data.size) / sf
    # plt.plot(time, data, lw=1.5, color='k', alpha=0.5)
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Voltage')
    # plt.xlim([time.min(), time.max()])
    # plt.title('N3 sleep EEG data (9)')
    # sns.despine()
    # plt.show()
    #
    # from scipy import signal
    #
    # # Define window length (4 seconds)
    # win = 4 * sf
    #
    # # Plot the power spectrum
    # sns.set(font_scale=1.2, style='white')
    # plt.figure(figsize=(8, 4))
    # for ww in range(0,5):
    #     data = np.array(ds1)[ww,:,4]
    #     freqs, psd = signal.welch(data, sf, nperseg=win)
    #     # psd = psd*psd
    #     plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
    # # data = np.mean(ds1,axis=0)[:,0]
    # # data = data*(np.concatenate((np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128), np.hanning(128)), axis=None))
    # freqs, psd = signal.welch(data, sf, nperseg=win)
    # plt.plot(freqs, psd, color='k', lw=1, alpha=0.5)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (V^2 / Hz)')
    # # plt.ylim([0, 2500])
    # plt.title("Welch's periodogram")
    # plt.xlim([0, freqs.max()])
    # sns.despine()
    # plt.show()

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
