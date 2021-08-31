import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
import time
from tkinter import TclError

# to display in separate Tk window
#matplotlib tk

# constants
CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
Fs=RATE

filter_length=4410

def IIR_filter_cutoff(cutoff_freq, filter_order=3, filter_type="lowpass"):
    try:
        w_c=2*cutoff_freq/Fs #cutoff freq in rad/s

        [num,den]= signal.butter(filter_order, w_c, btype= filter_type)

        #freq response
        [w,H]=signal.freqz(num, den, worN=filter_length)
        w=Fs*(w)/(2*np.pi)

        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w)
    except:
        print("Invalid Input")

def IIR_filter_zpk(zeros, poles, gain=1):
    try:
        #freq response
        [w,H]=signal.freqz_zpk(zeros, poles, gain, worN=filter_length)
        w=Fs*(w)/(2*np.pi)
        
        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w)
    except:
        print("invalid input")

def IIR_filter_rational(numerator, denominator=[1]):
    #freq response
    try:
        [w,H]=signal.freqz(numerator, denominator, worN=filter_length)
        w=Fs*(w)/(2*np.pi)

        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w)
    except:
        print("invalid input")

def FIR_filter_cutoff(cutoff_freq, num_taps, window_type="hamming", filter_type="lowpass"):
    try:
        w_c=2*cutoff_freq/Fs #cutoff freq in rad/s

        t=signal.firwin(num_taps,w_c,window=window_type, pass_zero=filter_type) #taps

        #freq response
        [w,H]=signal.freqz(t, worN=filter_length)
        w=Fs*(w)/(2*np.pi)

        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w)
    except:
        print("invalid input")

def FIR_filter_coefficients(coeff):
    try:
        #freq response
        [w,H]=signal.freqz(coeff, worN=filter_length)
        w=Fs*(w)/(2*np.pi)

        h=np.real(np.fft.ifft(H))               #impulse response

        #filtered = signal.convolve(input_signal, h, mode='same')

        return (h,H,w)
    except:
        print("invalid input")

def filter(input_signal, filter_impulse_response):
    try:
        filtered = signal.convolve(input_signal, filter_impulse_response, mode='same')
        return filtered
    except:
        print("Not able to filter")
        out=np.zeros(len(input_signal))
        return out

def plot_filter_mag_response(H,w):
    H_db=20*np.log10(abs(H))
    plt.plot(w,H_db)
    plt.title('Frequency Response')
    plt.xlabel('Frequency (in Hz)')
    plt.ylabel('Magnitude (in dB)')
    plt.grid('on')
    plt.figure()

def plot_filter_phase_response(H,w):
    w=2*np.pi*w/Fs
    H_arg=np.angle(H, deg=False)
    plt.plot(w,H_arg)
    plt.title('Frequency Response')
    plt.xlabel('Frequency (in rad)')
    plt.ylabel('Phase (in rad)')
    plt.grid('on')
    plt.figure()
    

def plot_filter_impulse_response(h):
    n=np.linspace(0,len(h),filter_length)
    plt.plot(n,h)
    plt.title('Impulse Response')
    plt.xlabel('Samples')
    plt.ylabel('Response')
    plt.figure()
    

def plot_filter(h,H,w):
    #plot_filter_impulse_response(h)
    #plot_filter_mag_response(H,w)
    #plot_filter_phase_response(H,w)
    
    fig1, ax=plt.subplots(2,2)
    n=np.linspace(0,len(h),filter_length)
    ax[0,0].plot(n,h)
    ax[0,0].set_title('Impulse Response')
    ax[0,0].set_xlabel('Samples')
    ax[0,0].set_ylabel('Response')

    w_r=2*np.pi*w/Fs
    H_arg=np.angle(H, deg=False)

    ax[0,1].plot(w_r, H_arg)
    ax[0,1].set_title('Phase Response')
    ax[0,1].set_xlabel('Frequency (in rad)')
    ax[0,1].set_ylabel('Phase (in rad)')
    ax[0,1].grid('on')
    
    H_db=20*np.log10(abs(H))

    ax[1,0].plot(w,H_db)
    ax[1,0].set_title('Magnitude Response')
    ax[1,0].set_xlabel('Frequency (in Hz)')
    ax[1,0].set_ylabel('Magnitude (in dB)')
    ax[1,0].grid('on')

    plt.figure()

def plot_signals(sig,filtered,t):
    plt.plot(t,sig, label="Input")
    plt.plot(t,filtered, label="Output")
    plt.title('Signals')
    plt.xlabel('Time(in s)')
    plt.ylabel('Amplitude')
    #plt.figure()


def let_user_pick(options):
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1,element))
    i = input("Enter number: ")
    try:
        if 0 < int(i) <= len(options):
            return int(i)
    except:
        pass
    return 1

t = np.linspace(0, 1,Fs)
u = (1*np.cos(2*np.pi*400*t) + 0.4*np.sin(2*np.pi*5000*t) + 0.01*np.cos(2*np.pi*10000*t))

filter_models=['IIR with cutoff frequency', 'IIR with zeros, poles and gain', 'IIR with numerator and denominator coefficients', 'FIR with cutoff frequency and window type', 'FIR with taps']
filter_types=['Lowpass', 'Highpass', 'Bandpass', 'Bandstop']
fir_windows=['Hamming', 'Hann', 'Blackman', 'Cosine']

#print("Choose type of filter and filter inputs:\n 1) IIR with cutoff frequency \n2) IIR with zeros, poles and gain \n3) IIR with numerator and denominator coefficients\n4) FIR with cutoff frequency and window type\n5) FIR with taps")
print("Choose type of filter and filter inputs:")
input_type=let_user_pick(filter_models)

if(input_type==1):
    #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
    print("Choose Type of Filter:")
    type_filter=let_user_pick(filter_types)
    if(type_filter==1):
        print("Input cutoff frequency (in Hz):")
        fc=np.array(input(), dtype='int')
        print("Input filter order:")
        order=int(input())
        filter_impulse_response, filter_freq_resp, frequencies=IIR_filter_cutoff(fc, order, filter_type="lowpass")
    elif(type_filter==2):
        print("Input cutoff frequency (in Hz):")
        fc=np.array(input(), dtype='int')
        print("Input filter order:")
        order=int(input())
        filter_impulse_response, filter_freq_resp, frequencies=IIR_filter_cutoff(fc, order, filter_type="highpass")
    elif(type_filter==3):
        print("Input lower cutoff frequency (in Hz):")
        fc1=int(input())
        print("Input upper cutoff frequency (in Hz):")
        fc2=int(input())
        print("Input filter order:")
        order=int(input())
        filter_impulse_response, filter_freq_resp, frequencies=IIR_filter_cutoff(np.array([fc1,fc2]), order, filter_type="bandpass")
    elif(type_filter==4):
        print("Input lower cutoff frequency (in Hz):")
        fc1=int(input())
        print("Input upper cutoff frequency (in Hz):")
        fc2=int(input())
        print("Input filter order:")
        order=int(input())
        filter_impulse_response, filter_freq_resp, frequencies=IIR_filter_cutoff(np.array([fc1,fc2]), order, filter_type="bandstop")   
elif(input_type==2):
    print("Input zeros:")
    z=np.array(input(), dtype='float')
    print("Input poles:")
    p=np.array(input(), dtype='float')
    print("Input gain:")
    k=float(input())
    filter_impulse_response, filter_freq_resp, frequencies=IIR_filter_zpk(z, p, k)
elif(input_type==3):
    print("Input Numerator coefficients for transfer function:")
    num=np.array(input(), dtype='float')
    print("Input Denominator coefficients for transfer function:")
    den=np.array(input(), dtype='float')
    filter_impulse_response, filter_freq_resp, frequencies=IIR_filter_rational(num, den)
elif(input_type==4):
    
    #print("ChooseType of window:\n1)Hamming\n2)Hann\n3)Blackman\n4)Cosine")
    print("ChooseType of window:")
    win=let_user_pick(fir_windows)
    if(win==1):
        #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
        print("Choose Type of Filter:")
        type_filter=let_user_pick(filter_types)
        if(type_filter==1):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="hamming", filter_type="lowpass")
        elif(type_filter==2):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="hamming", filter_type="highpass")
        elif(type_filter==3):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="hamming", filter_type="bandpass")
        elif(type_filter==4):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="hamming", filter_type="bandstop")
    elif(win==2):
        #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
        print("Choose Type of Filter:")
        type_filter=let_user_pick(filter_types)
        if(type_filter==1):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="hann", filter_type="lowpass")
        elif(type_filter==2):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="hann", filter_type="highpass")
        elif(type_filter==3):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="hann", filter_type="bandpass")
        elif(type_filter==4):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="hann", filter_type="bandstop")
    elif(win==3):
        #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
        print("Choose Type of Filter:")
        type_filter=let_user_pick(filter_types)
        if(type_filter==1):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="blackman", filter_type="lowpass")
        elif(type_filter==2):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="blackman", filter_type="highpass")
        elif(type_filter==3):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="blackman", filter_type="bandpass")
        elif(type_filter==4):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="blackman", filter_type="bandstop")
    elif(win==4):
        #print("Choose Type of Filter:\n 1) lowpass\n 2) highpass\n 3) bandpass\n 4) bandstop")
        print("Choose Type of Filter:")
        type_filter=let_user_pick(filter_types)
        if(type_filter==1):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="cosine", filter_type="lowpass")
        elif(type_filter==2):
            print("Input cutoff frequency (in Hz):")
            fc=np.array(input(), dtype='int')
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(fc, num, window_type="cosine", filter_type="highpass")
        elif(type_filter==3):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="cosine", filter_type="bandpass")
        elif(type_filter==4):
            print("Input lower cutoff frequency (in Hz):")
            fc1=int(input())
            print("Input upper cutoff frequency (in Hz):")
            fc2=int(input())
            print("Input the order number of taps:")
            num=int(input())
            filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_cutoff(np.array([fc1,fc2]), num, window_type="cosine", filter_type="bandstop")
elif(input_type==5):
    print("Input the taps:")
    t=np.array(input(), dtype='float')
    filter_impulse_response, filter_freq_resp, frequencies=FIR_filter_coefficients(t)

f=filter(u, filter_impulse_response)

plot_filter(filter_impulse_response, filter_freq_resp, frequencies)
plot_signals(u,f,t)
plt.show()

print("---THE END---")