from metpy.units import units
G = 9.8                     #m * s^-2
SPATIAL_RESOLUTION = 5
HEIGHT_SAMPLING_FREQ = 1/SPATIAL_RESOLUTION      #1/m used in interpolating data height-wise
MIN_ALT = 1000 * units.m     #minimun altitude of analysis
P_0 = 1000 * units.hPa      #needed for potential temp calculatiion
N_TRIALS = 1000         #number of bootstrap iterations
#for butterworth filter
LOWCUT = 100  #m - lower vertical wavelength cutoff for Butterworth bandpass filter
HIGHCUT = 4000  #m - upper vertical wavelength cutoff for Butterworth bandpass filter
ORDER = 3   #Butterworth filter order - Dutta(2017)