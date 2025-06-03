import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a time domain signal (mix of sine waves)
t = np.linspace(0, 2*np.pi, 1000)
signal = np.sin(2*t) + 0.5*np.sin(5*t)  # Original signal

# Step 2: Compute the Fourier Transform
Y = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), d=(t[1] - t[0]))

# Step 3: Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Step 4: Plot the frequency domain
plt.subplot(2, 1, 2)
plt.stem(freq[:500], np.abs(Y[:500]))
plt.title('Fourier Transform (Frequency Domain)')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()
