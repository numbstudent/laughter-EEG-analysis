import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


# filename=['gav','ovi','ila','hrz','aldoh','cra','cips','vyn','rijuu','sinlo','skot','manai','nature']
filename=['ovi2']
# filename=[    'cra','cips','vyn','rijuu','sinlo','skot','manai','nature']
useica = 0
X1 = []
X2 = []
# for ww in range(0, len(filename)):
# for ww in range(0, len(filename)):
#     with open('Z:/nani/experiment/TN_short laugh 1.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#         d1 = pickle.load(f)
#     with open('Z:/nani/experiment/TN_short laugh 2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#         d2 = pickle.load(f)
#     if ww==0:
#         X1 = d1[0]
#         X2 = d2[0]
#     else:
#         X1 = np.append(X1,(d1[0]),axis=0)
#         X2 = np.append(X2,(d2[0]),axis=0)
# X1 = np.array(X1)
# X2 = np.array(X2)
#
# data = np.mean(X2,axis=0)
# channel = 4
# data = data[:,channel]
# print(data.shape)
sf = 128.

# Multitaper delta power
# bp = bandpower(data, sf, [30, 50], 'multitaper')
# bp_rel = bandpower(data, sf, [30, 50], 'multitaper', relative=True)
# print('Absolute delta power: %.3f' % bp)
# print('Relative delta power: %.3f' % bp_rel)





def plot_spectrum_methods(data, sf, window_sec, band=None, dB=False):
    """Plot the periodogram, Welch's and multitaper PSD.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds for Welch's PSD
    dB : boolean
        If True, convert the power to dB.
    """
    from mne.time_frequency import psd_array_multitaper
    from scipy.signal import welch, periodogram
    sns.set(style="white", font_scale=1.2)
    # Compute the PSD
    freqs, psd = periodogram(data, sf)
    freqs_welch, psd_welch = welch(data, sf, nperseg=window_sec*sf)
    psd_mt, freqs_mt = psd_array_multitaper(data, sf, adaptive=True,
                                            normalization='full', verbose=0)
    sharey = False
    return freqs, psd, freqs_welch, psd_welch, freqs_mt, psd_mt


# Example: plot the 0.5 - 2 Hz band
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
# 'ovi2',
'key'
]
band = [0.5, 50]
dB=True
jenis = 'funny laugh'
title = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

for channel in range(0,13):
    print('channel ',channel)
    freqs = []
    psd = []
    freqs_welch = []
    psd_welch = []
    freqs_mt = []
    psd_mt = []
    for yy in range(0,len(sn)):
        with open('Z:/nani/experiment/'+sn[yy]+'/'+jenis+'_yes.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            d1 = pickle.load(f)
        data = d1[0]
        # channel = 4
        data = np.array(data)
        # print(data.shape)
        channel = int(channel)
        petek = data[:,:,channel]
        # print(np.array(data).shape)
        # exit()
        for xx in range(0,len(data)):
            aa,bb,cc,dd,ee,ff = plot_spectrum_methods(petek[xx], sf, 5, band, dB)
            freqs.append(aa)
            psd.append(bb)
            freqs_welch.append(cc)
            psd_welch.append(dd)
            freqs_mt.append(ee)
            psd_mt.append(ff)

        freqs = np.mean(freqs,axis=0)
        psd = np.mean(psd,axis=0)
        freqs_welch = np.mean(freqs_welch,axis=0)
        psd_welch = np.mean(psd_welch,axis=0)
        freqs_mt = np.mean(freqs_mt,axis=0)
        psd_mt = np.mean(psd_mt,axis=0)

        # Optional: convert power to decibels (dB = 10 * log10(power))
        if dB:
            psd = 10 * np.log10(psd)
            psd_welch = 10 * np.log10(psd_welch)
            psd_mt = 10 * np.log10(psd_mt)
            sharey = True

        # Start plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
        # Stem
        sc = 'red'
        ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
        ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
        ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
        # Line
        lc, lw = 'k', 2
        ax1.plot(freqs, psd, lw=lw, color=lc)
        ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
        ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)

        with open('Z:/nani/experiment/'+sn[yy]+'/'+jenis+'_no.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            d1 = pickle.load(f)
        data = d1[0]
        # channel = 4
        data = np.array(data)
        # print(data.shape)
        channel = int(channel)
        petek = data[:,:,channel]
        # print(np.array(data).shape)
        # exit()
        for xx in range(0,len(data)):
            aa,bb,cc,dd,ee,ff = plot_spectrum_methods(petek[xx], sf, 5, band, dB)
            freqs.append(aa)
            psd.append(bb)
            freqs_welch.append(cc)
            psd_welch.append(dd)
            freqs_mt.append(ee)
            psd_mt.append(ff)

        freqs = np.mean(freqs,axis=0)
        psd = np.mean(psd,axis=0)
        freqs_welch = np.mean(freqs_welch,axis=0)
        psd_welch = np.mean(psd_welch,axis=0)
        freqs_mt = np.mean(freqs_mt,axis=0)
        psd_mt = np.mean(psd_mt,axis=0)

        # Optional: convert power to decibels (dB = 10 * log10(power))
        if dB:
            psd = 10 * np.log10(psd)
            psd_welch = 10 * np.log10(psd_welch)
            psd_mt = 10 * np.log10(psd_mt)
            sharey = True

        # Start plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
        # Stem
        sc = 'slategrey'
        ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
        ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
        ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
        # Line
        lc, lw = 'k', 2
        ax1.plot(freqs, psd, lw=lw, color=lc)
        ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
        ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)

        # Labels and axes
        ax1.set_xlabel('Frequency (Hz)')
        ax2.set_xlabel('Frequency (Hz)')
        ax3.set_xlabel('Frequency (Hz)')
        if not dB:
            ax1.set_ylabel('Power spectral density (V^2/Hz)')
        else:
            ax1.set_ylabel('Decibels (dB / Hz)')
        ax1.set_title('Periodogram')
        ax2.set_title('Welch')
        ax3.set_title('Multitaper')
        if band is not None:
            ax1.set_xlim(band)
        ax1.set_ylim(ymin=0,ymax=50)
        ax2.set_ylim(ymin=0,ymax=50)
        ax3.set_ylim(ymin=0,ymax=50)
        sns.despine()
        plt.title(title[channel])
        # plt.show()
        plt.savefig('whihi '+jenis+' ch'+str(channel)+'.jpg')
        plt.close()
