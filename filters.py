import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# plt.close("all")

# Designing the IIR Butterworth filter
Fs = 10000  # sampling frequency
n = 9  # filter order

def bandpass_filter():
    fc = np.array([1000, 3000])  # cutoff frequency

    w_c = 2 * fc / Fs  # normalized frequency
    [b, a] = sig.butter(n, w_c, btype="bandpass")

    # frequency response
    [w, h] = sig.freqz(b, a, worN=2000)
    w = Fs * w / (2 * pi)

    h_db = 20 * np.log10(abs(h))

    plt.figure(1)
    plt.plot(w, h_db)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Magnitude(dB)")
    plt.grid(True)
    plt.show()


def highpass_filter():
    fc = 10000  # cutoff frequency

    w_c = 2 * fc / Fs  # normalized frequency
    [b, a] = sig.butter(n, w_c, btype="highpass")

    # frequency response
    [w, h] = sig.freqz(b, a, worN=2000)
    w = Fs * w / (2 * pi)

    h_db = 20 * np.log10(abs(h))

    plt.figure(1)
    plt.plot(w, h_db)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Magnitude(dB)")
    plt.grid(True)
    plt.show()


def lowpass_filter():
    fc = 1000  # cutoff frequency

    w_c = 2 * fc / Fs  # normalized frequency
    [b, a] = sig.butter(n, w_c, btype="low")

    # frequency response
    [w, h] = sig.freqz(b, a, worN=2000)
    w = Fs * w / (2 * pi)

    h_db = 20 * np.log10(abs(h))

    plt.figure(1)
    plt.plot(w, h_db)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Magnitude(dB)")
    plt.grid(True)
    plt.show()
