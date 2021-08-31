import matplotlib.pyplot as plt  # For plotting
from math import sin, pi  # For generating input signals


### Filter - Bandpass Filter
### @param [in] input - input unfiltered signal
### @param [out] output - output filtered signal
def bandpass_filter(x):
    y = [0] * 48000
    for n in range(4, len(x)):
        y[n] = 0.0101 * x[n] - 0.0202 * x[n - 2] + 0.0101 * x[n - 4] + 2.4354 * y[n - 1] - 3.1869 * y[n - 2] + 2.0889 * \
               y[n - 3] - 0.7368 * y[n - 4]
    return y


def highpass_filter(x):
    y = [0] * 48000
    for n in range(4, len(x)):
        y[n] = 0.0101 * x[n] + 2.4354 * y[n - 1] - 3.1869 * y[n - 2] + 2.0889 * y[n - 3] - 0.7368 * y[n - 4]
    return y


def lowpass_filter(x):
    y = [0] * 48000
    for n in range(4, len(x)):
        y[n] = 0.0101 * x[n] + 0.0202 * x[n - 1]
    return y


frequency = int(input("Please input the frequency: "))
print("Types of filters:\n"
      "1. Lowpass\n"
      "2. Highpass\n"
      "3. Bandpass\n")
type_fil = input("Please select the number for the type of filter:")

# Create empty arrays
input_sig = [float(0)] * 48000
output = [0] * 48000


# Fill array with signal
for i in range(48000):
    input_sig[i] = sin((2 * pi * frequency * i) / 48000)

# giving the signal as input
if type_fil == 1:
    output = lowpass_filter(input_sig)
elif type_fil == 2:
    output = highpass_filter(input_sig)
elif type_fil == 3:
    output = bandpass_filter(input_sig)


# input and output #1/100th of a second
output_section = output[0:480]
input_section = input_sig[0:480]

# Plot the signals for comparison
plt.figure(1)
plt.subplot(211)
plt.ylabel('Magnitude')
plt.xlabel('Samples')
plt.title('Unfiltered Signal')
plt.plot(input_section)
plt.subplot(212)
plt.ylabel('Magnitude')
plt.xlabel('Samples')
plt.title('Filtered Signal')
plt.plot(output_section)
plt.show()
