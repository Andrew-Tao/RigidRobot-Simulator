import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# Define the transfer function
num = np.array([0, 1])

den = np.array([1, 0.1, 1])
H = signal.TransferFunction(num, den)
# Set up the frequency range
wa = np.logspace(-1, 1, 5000,base=10)
# Calculate the frequency response
w, mag, phase = signal.bode(H,wa)
# Plot the Bode plot
fig, axs = plt.subplots(2, 1, figsize=(6, 9))
# Plot the magnitude response
axs[0].semilogx(w, mag)
axs[0].set_ylabel("Magnitude (dB)")
# Set the axis ranges
axs[0].set_xlim(0.1, 10) # Set the x-axis range from 0.1 to 10
axs[0].set_ylim(-40, 40) # Set the y-axis range from -40 to 40
# Plot the phase response
axs[1].semilogx(w, phase)
axs[1].set_ylabel("Phase (degrees)")
axs[1].set_xlabel("Frequency (rad/s)")
axs[1].set_xlim(0.1, 10) # Set the x-axis range from 0 to 5
axs[1].set_ylim(-180, 90) # Set the y-axis range from 0 to 50
# Show the plot
plt.show()