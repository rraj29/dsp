import matplotlib.pyplot as plt #For plotting
from math import sin, pi #For generating input signals

### Filter - 6KHz->8Khz Bandpass Filter
### @param [in] input - input unfiltered signal
### @param [out] output - output filtered signal
def filter(x):
    y = [0]*48000
    for n in range(4, len(x)):
        y[n] = 0.0101*x[n] - 0.0202*x[n-2] + 0.0101*x[n-4] + 2.4354*y[n-1] - 3.1869*y[n-2] + 2.0889*y[n-3] - 0.7368*y[n-4]
    return y


frequency = int(input("Please input the frequency: "))
	
### Create empty arrays
input = [0]*48000
output = [0]*48000

### Fill array with xxxHz signal
for i in range(48000):
    input[i] = sin(2 * pi * frequency * i / 48000) #+ sin(2 * pi * 70 * i / 48000)

### Run the signal through the filter
output = filter(input)

### Grab samples from input and output #1/100th of a second
output_section = output[0:480]  
input_section = input[0:480] 

### Plot the signals for comparison
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