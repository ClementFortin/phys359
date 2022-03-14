
# Doing the calibration with Helium and Neon

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Helium
# Helium spectral lines (best guesses) in nm and the corresponding region in dial
HeSpectra = [388.86, 447.12, 471.31, 492.19, 501.57, 587.56]
HeRegions = [[58.25, 58.75], [67.1, 67.55], [70.65, 71.2], [73.85, 74.3], [75.3, 75.7], [88.35, 88.7]]

# Load the data - Change the path as needed
path = r'/Users/selimamar/PycharmProjects/AtomicSpectra/Data/HeliumCalibration/He_full_scan_closed_11V_SlowScan.data'
data = np.loadtxt(path)

'''# Extract the 1D arrays of interest
# Finding where the dial reading is at x0 and x1 in order to get the range
x0, x1 = np.where(data[:,0] == 75.3)[0][0], np.where(data[:,0] == 75.7)[0][0]
dial = data[:,0][x1: x0]
V    = data[:,1][x1: x0]
std  = data[:,2][x1: x0] # Do not trust this quantity

# Plot the data
plt.errorbar(dial, V)
plt.xlabel('Dial Ticks')
plt.ylabel('PMT Voltage Mean (V)')
plt.savefig('Calibration Data for Helium')'''

# Defining a function to find the Gaussian peaks
def Gaussian(x,A,x0,sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def GaussianFit(regions):
    locations = np.zeros((len(regions),2))
    for idx, interval in enumerate(regions):
        x0, x1 = np.where(data[:, 0] == interval[0])[0][0], np.where(data[:, 0] == interval[1])[0][0]
        dial = data[:, 0][x1: x0]
        V = data[:, 1][x1: x0]
        std = data[:, 2][x1: x0]
        popt, pcov = curve_fit(Gaussian, dial, V, p0 = [100, np.mean(interval), 2], sigma = std)
        perr = np.sqrt(np.diag(pcov))
        locations[idx, 0], locations[idx, 1] = popt[1], perr[1]

    return locations

locations = GaussianFit(HeRegions)
print(locations)

# Doing the calibration
def lin_cal(x, m, b):
    return x/m - b/m

popt, pcov = curve_fit(lin_cal, HeSpectra, locations[:,0], p0 = [6.65, 0], sigma = locations[:,1])
perr = np.sqrt(np.diag(pcov))
mHe, bHe = [popt[0], perr[0]], [popt[1], perr[1]]
print(mHe, bHe)

# Plotting it
plt.plot(np.linspace(55, 95, 1000), mHe[0] * np.linspace(55, 95, 1000) + bHe[0])
plt.scatter(locations[:,0], HeSpectra, c='red')
#plt.errorbar(locations[:,0], HeSpectra, xerr = locations[:,1], fmt='o')
plt.xlabel('Dial Ticks')
plt.ylabel('Wavelength (nm)')
plt.show()

