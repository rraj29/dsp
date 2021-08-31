################################################################################################################
import pyaudio
from tkinter import *
from tkinter.ttk import Combobox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
import time

###############################################################################################################
filter_impulse_response = []
filter_freq_resp = []
frequencies = []
zeros = []
poles = []
z = [1]
p = [0.5]
k = 1
f = 100
f1 = 1000
f2 = 2000
o = 2
n = 1
d = 1
c = 1
win = "hamming"
widgets = []
widgets2 = []
widgets3 = []

###############################################################################################################

CHUNK = 1024 * 2  # samples per frame
FORMAT = pyaudio.paInt16  # audio format (bytes per sample?)
CHANNELS = 1  # single channel for microphone
RATE = 44100  # samples per second
Fs = RATE

filter_length = 512


###############################################################################################################

def IIR_filter_cutoff(cutoff_freq, filter_order, filter_type="lowpass"):
    global filter_impulse_response, filter_freq_resp, frequencies, zeros, poles
    w_c = cutoff_freq / Fs  # cutoff freq in rad/s

    [num, den] = signal.butter(int(filter_order), w_c, btype=filter_type)

    z = np.poly1d(num).roots
    p = np.poly1d(den).roots
    # freq response
    [w, H] = signal.freqz(num, den, worN=filter_length)
    w = Fs * (w) / (np.pi)

    h = np.real(np.fft.ifft(H))  # impulse response
    H1 = fft(h)
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles = (h, H1, w, z, p)


def IIR_filter_zpk(z, p, gain):
    global filter_impulse_response, filter_freq_resp, frequencies, zeros, poles
    # freq response
    [w, H] = signal.freqz_zpk(z, p, gain, worN=filter_length)
    w = Fs * (w) / (np.pi)

    h = np.real(np.fft.ifft(H))  # impulse response
    H1 = fft(h)
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles = (h, H1, w, z, p)


def IIR_filter_rational(numerator, denominator=[1]):
    global filter_impulse_response, filter_freq_resp, frequencies, zeros, poles
    [w, H] = signal.freqz(numerator, denominator, worN=filter_length)
    w = Fs * (w) / (np.pi)
    z = np.poly1d(numerator).roots
    p = np.poly1d(denominator).roots
    h = np.real(np.fft.ifft(H))  # impulse response
    H1 = fft(h)
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles = (h, H1, w, z, p)


def FIR_filter_cutoff(cutoff_freq, num_taps, window_type="hamming", filter_type="lowpass"):
    global filter_impulse_response, filter_freq_resp, frequencies, zeros, poles
    w_c = cutoff_freq / Fs  # cutoff freq in rad/s

    t = signal.firwin(int(num_taps), w_c, window=window_type, pass_zero=filter_type)  # taps
    z = np.poly1d(t).roots
    p = []
    # freq response
    [w, H] = signal.freqz(t, worN=filter_length)
    w = Fs * (w) / (np.pi)

    h = np.real(np.fft.ifft(H))  # impulse response
    H1 = fft(h)
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles = (h, H1, w, z, p)


def FIR_filter_coefficients(coeff):
    global filter_impulse_response, filter_freq_resp, frequencies, zeros, poles
    # freq response
    [w, H] = signal.freqz(coeff, worN=filter_length)
    w = Fs * (w) / (np.pi)
    z = np.poly1d(coeff).roots
    p = []
    h = np.real(np.fft.ifft(H))  # impulse response
    H1 = fft(h)
    filter_impulse_response, filter_freq_resp, frequencies, zeros, poles = (h, H1, w, z, p)


def filter(input_signal, filter_impulse_response):
    try:
        filtered = np.convolve(input_signal, filter_impulse_response, mode='same')
        return filtered
    except:
        print("Not able to filter")
        out = np.zeros(len(input_signal))
        return out


def plot_filter(h, H, w, z, p):
    fig1, ax = plt.subplots(2, 2)
    n1 = np.linspace(0, len(h), filter_length)
    ax[0, 0].plot(n1, h)
    ax[0, 0].set_title('Impulse Response')
    ax[0, 0].set_xlabel('Samples')
    ax[0, 0].set_ylabel('Response')

    w_r = 2 * np.pi * w / Fs
    H_arg = np.angle(H, deg=False)

    ax[0, 1].plot(w_r[0:int(len(w_r) / 2)], H_arg[0:int(len(H_arg) / 2)])
    ax[0, 1].set_title('Phase Response')
    ax[0, 1].set_xlabel('Frequency (in rad)')
    ax[0, 1].set_ylabel('Phase (in rad)')
    ax[0, 1].grid('on')

    H_db = 20 * np.log10(abs(H))

    ax[1, 0].plot(w[0:int(len(w) / 2)], H_db[0:int(len(H_db) / 2)])
    ax[1, 0].set_title('Magnitude Response')
    ax[1, 0].set_xlabel('Frequency (in Hz)')
    ax[1, 0].set_ylabel('Magnitude (in dB)')
    ax[1, 0].grid('on')

    theta = np.linspace(0, 2 * np.pi, 100)
    ejtheta = np.exp(1j * theta)
    ax[1, 1].plot(np.real(ejtheta), np.imag(ejtheta))
    for pt in z:
        ax[1, 1].plot(np.real(pt), np.imag(pt), 'ro')
    for pt in p:
        ax[1, 1].plot(np.real(pt), np.imag(pt), 'rx')
    ax[1, 1].set_title('Pole-Zero Plot')
    ax[1, 1].grid()
    ax[1, 1].set_aspect('equal', adjustable="datalim")

    plt.tight_layout()
    plt.show(block=False)


#########################################################################################################################################
def stream_start():
    global filter_impulse_response, filter_freq_resp
    fig3, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

    # pyaudio class instance
    p = pyaudio.PyAudio()

    player = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK
    )
    # stream object to get data from microphone
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # variable for plotting
    x = np.arange(0, 2 * CHUNK, 2)  # samples (waveform)
    xf = np.linspace(0, RATE, CHUNK)  # frequencies (spectrum)

    # create a line object with random data
    line, = ax1.plot(x, np.random.rand(CHUNK), label='Input', animated=True)
    line_filtered, = ax1.plot(x, np.random.rand(CHUNK), label='Output', animated=True)

    # create semilogx line for spectrum
    line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), label='Input', animated=True)
    line_fft_filtered, = ax2.semilogx(xf, np.random.rand(CHUNK), label='Output', animated=True)

    # format waveform axes
    ax1.set_title('AUDIO WAVEFORM')
    ax1.set_xlabel('samples')
    ax1.set_ylabel('volume')
    # ax1.set_ylim(-255, 256)
    ax1.set_xlim(0, 2 * CHUNK)
    ax1.legend()
    plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-255, 0, 256])

    # format spectrum axes
    ax2.set_xlim(20, RATE / 2)
    ax2.legend()

    plt.show(block=False)

    plt.pause(0.05)

    bg = fig3.canvas.copy_from_bbox(fig3.bbox)

    ax1.draw_artist(line)
    ax1.draw_artist(line_filtered)
    ax2.draw_artist(line_fft)
    ax2.draw_artist(line_fft_filtered)

    fig3.canvas.blit(fig3.bbox)

    print('stream started')

    # for measuring frame rate
    frame_count = 0
    start_time = time.time()

    while True:

        fig3.canvas.restore_region(bg)
        # binary data
        data = stream.read(CHUNK)

        data_int = np.frombuffer(data, dtype=np.int16)

        m = np.iinfo(np.int16)
        m1 = m.max

        data_np = 512 * ((data_int) / float(m1))

        line.set_ydata(data_np)

        filtered_int = filter(data_int, filter_impulse_response)
        filtered_np = 2 * (512 * (filtered_int)) / float(m1)

        line_filtered.set_ydata(filtered_np)

        yf = fft(data_int)
        line_fft.set_ydata(np.abs(yf[0:CHUNK]) / (m1 / 50 * CHUNK))

        fft_filtered = fft(filtered_int)
        line_fft_filtered.set_ydata(np.abs(fft_filtered[0:CHUNK]) / (m1 / 40 * CHUNK))

        filtered = np.array(filtered_int, dtype=np.int16)
        player.write(filtered, CHUNK)

        ax1.draw_artist(line)
        ax1.draw_artist(line_filtered)
        ax2.draw_artist(line_fft)
        ax2.draw_artist(line_fft_filtered)

        # update figure canvas
        if len(plt.get_fignums()) != 0:
            # fig3.canvas.draw()
            fig3.canvas.blit(fig3.bbox)
            fig3.canvas.flush_events()
            frame_count += 1

        else:

            # calculate average frame rate
            frame_rate = frame_count / (time.time() - start_time)

            print('stream stopped')
            print('average frame rate = {:.0f} FPS'.format(frame_rate))
            break

    p.terminate()
    # print("---THE END---")


#########################################################################################################################################

mainWindow = Tk()
mainWindow.title('Audio Filters')
mainWindow.geometry("520x620+10+10")

##########################################################################################################################################
zvar = StringVar()
pvar = StringVar()
kvar = StringVar()
fvar = StringVar()
f1var = StringVar()
f2var = StringVar()
ovar = StringVar()
dvar = StringVar()
cvar = StringVar()
nvar = StringVar()
winvar = StringVar()
filvar = StringVar()
typeVar = StringVar()


#########################################################################################################################################
def getEntry_fn():
    global fvar, ovar, winvar, f, o, win
    f = float(fvar.get())
    o = float(ovar.get())
    win = str(winvar.get())


def getEntry_c():
    global cvar, c
    c1 = cvar.get().split()
    c = np.array(c1, dtype="complex")


def getEntry_zpk():
    global zvar, pvar, kvar, z, p, k

    z = zvar.get().split()
    z = np.array(z, dtype="complex")
    p = pvar.get().split()
    p = np.array(p, dtype="complex")
    k = float(kvar.get())


def getEntry_fo():
    global fvar, ovar, f, o
    f = float(fvar.get())
    o = float(ovar.get())


def getEntry_ffo():
    global f1var, f2var, ovar, f1, f2, o
    f1 = float(f1var.get())
    f2 = float(f2var.get())
    o = float(ovar.get())


def getEntry_nd():
    global nvar, dvar, n, d
    n = nvar.get().split()
    n = np.array(n, dtype="complex")
    d = dvar.get().split()
    d = np.array(d, dtype="complex")


#################################################################################################################################

def create_fir():
    global widgets, widgets2, widgets3
    global en1
    for widget in widgets[:]:
        widget.destroy()
        widgets.remove(widget)

    lbl = Label(mainWindow, text="Select Input Method:", fg='blue', font=("Helvetica", 10))
    lbl.place(x=100, y=90)
    options = ("Cutoff Frequency(in Hz) and window Type", "Coefficients")
    cb = Combobox(mainWindow, values=options, width=43, textvariable=typeVar)
    cb.place(x=100, y=110)
    cb.bind("<<ComboboxSelected>>", firOptions)
    en1 = cb
    widgets = widgets[:] + [lbl, cb] + widgets2[:] + widgets3[:]


def create_iir():
    global widgets, widgets2, widgets3
    global en1
    for widget in widgets[:]:
        widget.destroy()
        widgets.remove(widget)

    lbl = Label(mainWindow, text="Select Input Method:", fg='blue', font=("Helvetica", 10))
    lbl.place(x=100, y=90)
    options = ("Poles and Zeros", "Cutoff Frequency(in Hz) and filter Order", "Numerator and denominator Coefficients")
    cb = Combobox(mainWindow, values=options, width=43, textvariable=typeVar)
    cb.place(x=100, y=110)
    cb.bind("<<ComboboxSelected>>", iirOptions)
    en1 = cb
    widgets = widgets[:] + [lbl, cb] + widgets2[:] + widgets3[:]


##########################################################################################################################################

def firOptions(event):
    global fvar, f1var, f2var, ovar, cvar, winvar, f, f1, f2, o, c, win, typeVar
    global widgets2, widgets3
    global en1
    if int(en1.current()) == 1:
        for widget in widgets2[:]:
            widget.destroy()
            widgets2.remove(widget)

        lblc = Label(mainWindow, text="Enter Coefficients (number with spaces):", fg='blue', font=("Helvetica", 10))
        lblc.place(x=90, y=170)
        c_ = Entry(mainWindow, textvariable=zvar)
        c_.delete(0, 'end')
        c_.insert(END, str(1))

        c_.place(x=100, y=200)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_c)
        btn1.place(x=100, y=230)

        btn = Button(mainWindow, text="Get Filter Response!", command=lambda: FIR_filter_coefficients(c))
        btn.place(x=100, y=260)

        widgets2 = widgets2[:] + [c_, btn, btn1, lblc] + widgets3[:]
    elif int(en1.current()) == 0:
        for widget in widgets2[:]:
            widget.destroy()
            widgets2.remove(widget)

        lbl = Label(mainWindow, text="Select One:", fg='blue', font=("Helvetica", 10))
        lbl.place(x=50, y=170)

        v0 = IntVar()
        v0.set(1)
        r1 = Radiobutton(mainWindow, text="Lowpass", variable=v0, value=1, command=lambda: give_options1(1))
        r2 = Radiobutton(mainWindow, text="Highpass", variable=v0, value=2, command=lambda: give_options1(2))
        r3 = Radiobutton(mainWindow, text="Bandpass", variable=v0, value=3, command=lambda: give_options1(3))
        r4 = Radiobutton(mainWindow, text="Bandstop", variable=v0, value=4, command=lambda: give_options1(4))
        r1.place(x=50, y=200)
        r2.place(x=120, y=200)
        r3.place(x=190, y=200)
        r4.place(x=260, y=200)

        widgets2 = widgets2[:] + [r1, r2, r3, r4, lbl] + widgets3[:]


def iirOptions(event):
    global zvar, pvar, kvar, fvar, f1var, f2var, ovar, nvar, dvar, z, p, k, f, f1, f2, o, d, typeVar
    global widgets2, widgets3
    global en1
    if int(en1.current()) == 0:
        for widget in widgets2[:]:
            widget.destroy()
            widgets2.remove(widget)

        lblz = Label(mainWindow, text="Enter Zeros (numbers with spaces):", fg='blue', font=("Helvetica", 10))
        lblz.place(x=90, y=170)
        zeros_ = Entry(mainWindow, textvariable=zvar)
        zeros_.delete(0, 'end')
        zeros_.insert(END, str(1))

        lblp = Label(mainWindow, text="Enter Poles (numbers with spaces):", fg='blue', font=("Helvetica", 10))
        lblp.place(x=90, y=230)
        poles_ = Entry(mainWindow, textvariable=pvar)
        poles_.delete(0, 'end')
        poles_.insert(END, str(0.5))

        lblk = Label(mainWindow, text="Enter Gain:", fg='blue', font=("Helvetica", 10))
        lblk.place(x=90, y=290)
        gain_ = Entry(mainWindow, textvariable=kvar)
        gain_.delete(0, 'end')
        gain_.insert(END, str(1))

        zeros_.place(x=100, y=200)
        poles_.place(x=100, y=260)
        gain_.place(x=100, y=320)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_zpk)
        btn1.place(x=100, y=350)

        btn = Button(mainWindow, text="Get Filter Response!", command=lambda: IIR_filter_zpk(z, p, k))
        btn.place(x=100, y=380)

        widgets2 = widgets2[:] + [zeros_, poles_, gain_, btn, btn1, lblz, lblp, lblk, btn1] + widgets3[:]

    elif int(en1.current()) == 1:
        for widget in widgets2[:]:
            widget.destroy()
            widgets2.remove(widget)
        lbl = Label(mainWindow, text="Select One:", fg='blue', font=("Helvetica", 10))
        lbl.place(x=50, y=140)

        v0 = IntVar()
        v0.set(1)
        r1 = Radiobutton(mainWindow, text="Lowpass", variable=v0, value=1, command=lambda: give_options(1))
        r2 = Radiobutton(mainWindow, text="Highpass", variable=v0, value=2, command=lambda: give_options(2))
        r3 = Radiobutton(mainWindow, text="Bandpass", variable=v0, value=3, command=lambda: give_options(3))
        r4 = Radiobutton(mainWindow, text="Bandstop", variable=v0, value=4, command=lambda: give_options(4))
        r1.place(x=50, y=170)
        r2.place(x=120, y=170)
        r3.place(x=190, y=170)
        r4.place(x=260, y=170)

        widgets2 = widgets2[:] + [r1, r2, r3, r4, lbl] + widgets3[:]

    elif int(en1.current()) == 2:
        for widget in widgets2[:]:
            widget.destroy()
            widgets2.remove(widget)

        lbln = Label(mainWindow, text="Input Numerator Coefficients of decreasing order of power of z:", fg='blue',
                     font=("Helvetica", 10))
        lbln.place(x=90, y=170)
        num_ = Entry(mainWindow, textvariable=nvar)
        num_.delete(0, 'end')
        num_.insert(END, str(1))

        lbld = Label(mainWindow, text="Input Denominator Coefficients of decreasing order of power of z:", fg='blue',
                     font=("Helvetica", 10))
        lbld.place(x=90, y=230)
        den_ = Entry(mainWindow, textvariable=dvar)
        den_.delete(0, 'end')
        den_.insert(END, str(1))

        num_.place(x=100, y=200)
        den_.place(x=100, y=260)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_nd)
        btn1.place(x=100, y=290)

        btn = Button(mainWindow, text="Get Filter Response!", command=lambda: IIR_filter_rational(n, d))
        btn.place(x=100, y=320)

        widgets2 = widgets2[:] + [num_, den_, btn, btn1, lbln, lbld] + widgets3[:]


############################################################################################################################################

def give_options(i):
    global zvar, pvar, kvar, fvar, f1var, f2var, ovar, nvar, dvar, z, p, k, f, f1, f2, o, n, d
    global widgets3

    if (i == 1):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblf = Label(mainWindow, text="Input Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf.place(x=90, y=200)
        fc_ = Entry(mainWindow, textvariable=fvar)
        fc_.delete(0, 'end')
        fc_.insert(END, str(1000))

        lblo = Label(mainWindow, text="Input filter order:", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=260)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        fc_.place(x=100, y=230)
        o_.place(x=100, y=290)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_fo)
        btn1.place(x=100, y=320)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: IIR_filter_cutoff(f, o, filter_type="lowpass"))
        btn.place(x=100, y=350)

        widgets3 = widgets3[:] + [fc_, o_, btn, btn1, lblo, lblf]
    elif (i == 2):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblf = Label(mainWindow, text="Input Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf.place(x=90, y=200)
        fc_ = Entry(mainWindow, textvariable=fvar)
        fc_.delete(0, 'end')
        fc_.insert(END, str(1000))

        lblo = Label(mainWindow, text="Input filter order:", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=260)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        fc_.place(x=100, y=230)
        o_.place(x=100, y=290)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_fo)
        btn1.place(x=100, y=320)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: IIR_filter_cutoff(f, o, filter_type="highpass"))
        btn.place(x=100, y=350)

        widgets3 = widgets3[:] + [fc_, o_, btn, btn1, lblo, lblf]
    elif (i == 3):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblf1 = Label(mainWindow, text="Input Lower Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf1.place(x=90, y=200)
        fc1_ = Entry(mainWindow, textvariable=f1var)
        fc1_.delete(0, 'end')
        fc1_.insert(END, str(1000))

        lblf2 = Label(mainWindow, text="Input Upper Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf2.place(x=90, y=260)
        fc2_ = Entry(mainWindow, textvariable=f2var)
        fc2_.delete(0, 'end')
        fc2_.insert(END, str(2000))

        lblo = Label(mainWindow, text="Input filter order:", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=320)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        fc1_.place(x=100, y=230)
        fc2_.place(x=100, y=290)
        o_.place(x=100, y=350)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_ffo)
        btn1.place(x=100, y=380)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: IIR_filter_cutoff(np.array([f1, f2]), o, filter_type="bandpass"))
        btn.place(x=100, y=410)

        widgets3 = widgets3[:] + [fc1_, fc2_, o_, btn, btn1, lblf1, lblf2, lblo]
    elif (i == 4):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblf1 = Label(mainWindow, text="Input Lower Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf1.place(x=90, y=200)
        fc1_ = Entry(mainWindow, textvariable=f1var)
        fc1_.delete(0, 'end')
        fc1_.insert(END, str(1000))

        lblf2 = Label(mainWindow, text="Input Upper Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf2.place(x=90, y=260)
        fc2_ = Entry(mainWindow, textvariable=f2var)
        fc2_.delete(0, 'end')
        fc2_.insert(END, str(2000))

        lblo = Label(mainWindow, text="Input filter order :", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=320)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        fc1_.place(x=100, y=230)
        fc2_.place(x=100, y=290)
        o_.place(x=100, y=350)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_ffo)
        btn1.place(x=100, y=380)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: IIR_filter_cutoff(np.array([f1, f2]), o, filter_type="bandstop"))
        btn.place(x=100, y=410)

        widgets3 = widgets3[:] + [fc1_, fc2_, o_, btn, btn1, lblf1, lblf2, lblo]


def give_options1(i):
    global fvar, f1var, f2var, ovar, winvar, f, f1, f2, o, win
    global widgets3

    if (i == 1):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblw = Label(mainWindow, text="Select Window type:", fg='blue', font=("Helvetica", 10))
        lblw.place(x=90, y=230)
        win_ = Combobox(mainWindow, textvariable=winvar, values=["hamming", "hann", "blackman", "cosine"])

        lblf = Label(mainWindow, text="Input Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf.place(x=90, y=290)
        fc_ = Entry(mainWindow, textvariable=fvar)
        fc_.delete(0, 'end')
        fc_.insert(END, str(1000))

        lblo = Label(mainWindow, text="Input number of taps:", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=350)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        win_.place(x=100, y=260)
        fc_.place(x=100, y=320)
        o_.place(x=100, y=380)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_fn)
        btn1.place(x=100, y=410)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: FIR_filter_cutoff(f, o, window_type=win, filter_type="lowpass"))
        btn.place(x=100, y=440)

        widgets3 = widgets3[:] + [fc_, o_, win_, btn, btn1, lblw, lblo, lblf]
    elif (i == 2):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblw = Label(mainWindow, text="Select Window type:", fg='blue', font=("Helvetica", 10))
        lblw.place(x=90, y=230)
        win_ = Combobox(mainWindow, textvariable=winvar, values=["hamming", "hann", "blackman", "cosine"])

        lblf = Label(mainWindow, text="Input Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf.place(x=90, y=290)
        fc_ = Entry(mainWindow, textvariable=fvar)
        fc_.delete(0, 'end')
        fc_.insert(END, str(1000))

        lblo = Label(mainWindow, text="Input number of taps (only odd number):", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=350)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        win_.place(x=100, y=260)
        fc_.place(x=100, y=320)
        o_.place(x=100, y=380)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_fn)
        btn1.place(x=100, y=410)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: FIR_filter_cutoff(f, o, window_type=win, filter_type="highpass"))
        btn.place(x=100, y=440)

        widgets3 = widgets3[:] + [fc_, o_, win_, btn, btn1, lblw, lblo, lblf]
    elif (i == 3):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblw = Label(mainWindow, text="Select Window type:", fg='blue', font=("Helvetica", 10))
        lblw.place(x=90, y=230)
        win_ = Combobox(mainWindow, textvariable=winvar, values=["hamming", "hann", "blackman", "cosine"])

        lblf1 = Label(mainWindow, text="Input Lower Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf1.place(x=90, y=290)
        fc1_ = Entry(mainWindow, textvariable=f1var)
        fc1_.delete(0, 'end')
        fc1_.insert(END, str(1000))

        lblf2 = Label(mainWindow, text="Input Upper Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf2.place(x=90, y=350)
        fc2_ = Entry(mainWindow, textvariable=f2var)
        fc2_.delete(0, 'end')
        fc2_.insert(END, str(2000))

        lblo = Label(mainWindow, text="Input number of taps:", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=410)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        win_.place(x=100, y=260)
        fc1_.place(x=100, y=320)
        fc2_.place(x=100, y=380)
        o_.place(x=100, y=440)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_ffo)
        btn1.place(x=100, y=470)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: FIR_filter_cutoff(np.array([f1, f2]), o, window_type=win, filter_type="bandpass"))
        btn.place(x=100, y=500)

        widgets3 = widgets3[:] + [fc1_, fc2_, o_, win_, btn, btn1, lblf1, lblf2, lblo, lblw]
    elif (i == 4):
        for widget in widgets3[:]:
            widget.destroy()
            widgets3.remove(widget)

        lblw = Label(mainWindow, text="Select Window type:", fg='blue', font=("Helvetica", 10))
        lblw.place(x=90, y=230)
        win_ = Combobox(mainWindow, textvariable=winvar, values=["hamming", "hann", "blackman", "cosine"])

        lblf1 = Label(mainWindow, text="Input Lower Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf1.place(x=90, y=290)
        fc1_ = Entry(mainWindow, textvariable=f1var)
        fc1_.delete(0, 'end')
        fc1_.insert(END, str(1000))

        lblf2 = Label(mainWindow, text="Input Upper Cutoff Frequency (in Hz):", fg='blue', font=("Helvetica", 10))
        lblf2.place(x=90, y=350)
        fc2_ = Entry(mainWindow, textvariable=f2var)
        fc2_.delete(0, 'end')
        fc2_.insert(END, str(2000))

        lblo = Label(mainWindow, text="Input number of taps (only odd number):", fg='blue', font=("Helvetica", 10))
        lblo.place(x=90, y=410)
        o_ = Entry(mainWindow, textvariable=ovar)
        o_.delete(0, 'end')
        o_.insert(END, str(2))

        win_.place(x=100, y=260)
        fc1_.place(x=100, y=320)
        fc2_.place(x=100, y=380)
        o_.place(x=100, y=440)

        btn1 = Button(mainWindow, text="Set Values!", command=getEntry_ffo)
        btn1.place(x=100, y=470)

        btn = Button(mainWindow, text="Get Filter Response!",
                     command=lambda: FIR_filter_cutoff(np.array([f1, f2]), o, window_type=win, filter_type="bandstop"))
        btn.place(x=100, y=500)

        widgets3 = widgets3[:] + [fc1_, fc2_, o_, win_, btn, btn1, lblf1, lblf2, lblo, lblw]


##########################################################################################################################################

lbl = Label(mainWindow, text="Select One Type:", fg='blue', font=("Helvetica", 10))
lbl.place(x=15, y=15)

v0 = IntVar()
v0.set(1)
r1 = Radiobutton(mainWindow, text="IIR Filter", variable=v0, value=1, command=create_iir)
r2 = Radiobutton(mainWindow, text="FIR Filter", variable=v0, value=2, command=create_fir)
r1.place(x=100, y=45)
r2.place(x=180, y=45)

fplot = Button(mainWindow, text="Plot Filter Response and zeros/poles",
               command=lambda: plot_filter(filter_impulse_response, filter_freq_resp, frequencies, zeros, poles))
fplot.place(x=150, y=550)

stream = Button(mainWindow, text="Start Streaming", command=stream_start)
stream.place(x=150, y=580)

#############################################################################################################################################

mainWindow.mainloop()

#############################################################################################################################################
