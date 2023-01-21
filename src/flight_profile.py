
#dependencies
import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#for ellipse fitting
#from math import atan2
#from numpy.linalg import eig, inv, svd

#data smoothing
from scipy import signal
from scipy import optimize

#metpy related dependencies - consider removing entirely
import metpy.calc as mpcalc
from metpy.units import units
from src import constants
class FlightProfiler():
    
    def __init__(self,applyButterworth:bool, p_0: units.hPa , heightSamplingFreq : units.m ,filepath) -> None:
        self.__applyButterworth = applyButterworth
        self.__p_0 = p_0
        self.__heightSamplingFreq = heightSamplingFreq
        # df 
        # self.__U_total, self.__V_total, self.__T_total
        self.__flight_data , self.dataframe = self.preprocessDataResample(filepath)
        #self.flightNumber = self.getFlightConfiguration(file, configFile, configPath)
    
    def __apply_butterworth(self,spatialResolution):
            #linear interpolation only needs to occur if butterworth is applied
        #linearly interpolate data - such that it is spaced iniformly in space, heightwise - stolen from Keaton
        #create index of heights with 1 m spacial resolution - from minAlt to maxAlt
        heightIndex = pd.DataFrame({'Alt': np.arange(min(df['Alt']), max(df['Alt']))})
        #right merge data with index to keep all heights
        df= pd.merge(df, heightIndex, how='right', on='Alt')
        #sort data by height
        df = df.sort_values(by='Alt')
        #linear interpolate the nans
        missingDataLimit = 999  #more than 1km of data should be left as nans, will not be onsidered in analysis
        df = df.interpolate(method='linear', limit=missingDataLimit)
        #resample at height interval
        keepIndex = np.arange(0, len(df['Alt']), spatialResolution)
        df = df.iloc[keepIndex,:]
        df.reset_index(drop=True, inplace=True)
        df = df.dropna()    #added 8/4/2021 - not sure why butterworth is creating nans

    # TODO: change to allow for different naming conventions 
    def getFlightConfiguration(profile, configFile, configFilePath):
        """retrieve information from site configuration file which contains site initials, flight times, tropopause heights
        """
        config = pd.read_csv(os.path.join(configFilePath, configFile), skiprows=[1], parse_dates=[[2,3]])
        num = profile.split("_")[0]     #get site initial and flight number from file name
        num = [x for x in num if x.isdigit()]   #remove site initial(s)
        num = int("".join(num))     #flight number by itself
        return num

    def __butter_bandpass_filter(self,data, lowcut, highcut, fs, order):
        """Applies Butterworth filter to perturbation profiles
        """
        #b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        #y = signal.lfilter(b, a, data)
        y = signal.sosfilt(sos, data)
        return y
    
    def __brunt_viasala(self ,Temp,Alt,Pres,u,v):
                #calculate brunt-viasala frequency **2 
        tempK = Temp.to('kelvin')
        potentialTemperature =  tempK * (self.__p_0 / Pres) ** (2/7)    #https://glossary.ametsoc.org/wiki/Potential_temperature   
        bv2 = mpcalc.brunt_vaisala_frequency_squared(Alt, potentialTemperature)   #N^2 
        bv = mpcalc.brunt_vaisala_frequency(Alt, potentialTemperature)
        print("BV MEAN: ", np.nanmean(bv))
        print("BV Period [min]: ", (2 * np.pi)/(np.nanmean(bv.magnitude) * 60))
        bv2 = bv2.magnitude 
        
        plt.scatter(Alt,bv2, s=0.75)
        meanBV2 = np.ones(len(bv2)) * np.mean(bv2)
        # localmean =
        plt.plot(Alt, meanBV2, color='red')
        #bv2 = bruntViasalaFreqSquared(potentialTemperature, heightSamplingFreq)     #Maybe consider using metpy version of N^2 ? Height sampling is not used in hodo method, why allow it to affect bv ?
                #Save raw (sum of background and perturbations) wind, temp data, in case needed in the furure
        U_total = u.magnitude
        V_total = v.magnitude
        T_total = tempK.magnitude
        return U_total,V_total,T_total

    def perterbation_profile(self,file, spatialResolution, lambda1, lambda2, order,Alt,Temp):
        #subtract nth order polynomials to find purturbation profile
        #detrend  temperature using polynomial fit
        '''
        Fig, axs = plt.subplots(2,4,figsize=(6,6), num=1)   #figure for temperature
        Fig2, axs2 = plt.subplots(2,4,figsize=(6,6), num=2)   #figure for wind
        axs = axs.flatten() #make subplots iteratble by single indice
        axs2 = axs2.flatten()
        '''
        temp_background = []
        u_background = []
        v_background = []

        for k in range(2,10):
            i = k-2
            
            #temp
            poly = np.polyfit(Alt.magnitude / 1000, Temp.magnitude, k)
            tempFit = np.polyval(poly, Alt.magnitude / 1000)
            temp_background.append(tempFit)
            
            #u
            poly = np.polyfit(Alt.magnitude / 1000, u.magnitude, k)
            uFit = np.polyval(poly, Alt.magnitude / 1000)
            u_background.append(uFit)

            #v
            poly = np.polyfit(Alt.magnitude / 1000, v.magnitude, k)
            vFit = np.polyval(poly, Alt.magnitude / 1000)
            v_background.append(vFit)
            # TODO: [1]
        
        #subtract fits to produce various perturbation profiles
        tempPert = []
        global uPert
        uPert = []
        vPert = []
        
        for i, element in enumerate(temp_background):
            pert = np.subtract(Temp.magnitude, temp_background[i])
            tempPert.append(pert)
            pert = np.subtract(u.magnitude, u_background[i])
            uPert.append(pert)
            pert = np.subtract(v.magnitude, v_background[i])
            vPert.append(pert)
        #plot to double check subtraction
        Fig, axs = plt.subplots(2,2,figsize=(6,6), num=3, sharey=True)#, sharex=True)   #figure for u,v butterworth filter
        for i,element in enumerate(u_background):
            axs[0,0].plot(uPert[i], Alt.magnitude/1000, linewidth=0.5, label="Order: {}".format(str(i+2)))
            axs[1,0].plot(vPert[i], Alt.magnitude/1000, linewidth=0.5)   
        Fig.legend()
        Fig.suptitle("Wind Components; Background Removed, Filtered \n {}".format(file))
        axs[0,0].set_xlabel("Zonal Wind (m/s)")
        axs[1,0].set_xlabel("Meridional Wind (m/s)")
        axs[0,1].set_xlabel("Filtered Zonal Wind (m/s)")
        axs[0,1].set_xlim([-10,10])
        axs[1,1].set_xlim([-10,10])
        axs[0,0].set_xlim([-20,35])
        axs[1,0].set_xlim([-20,35])
        axs[1,1].set_xlabel("Filtered Meridional Wind (m/s)")
        axs[0,0].set_ylabel("Altitude (km)")
        axs[1,0].set_ylabel("Altitude (km)")
        axs[0,0].tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
        axs[0,1].tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
        #Apply Butterworth Filter
        if self.__applyButterworth:
            #filter using 3rd order butterworth - fs=samplerate (1/m)
            freq2 = 1/lambda1    #find cutoff freq 1/m
            freq1 =  1/lambda2    #find cutoff freq 1/m
            # Plot the frequency response for a few different orders.
            #b, a = butter_bandpass(freq1, freq2, heightSamplingFreq, order)
            sos = self.__butter_bandpass(freq1, freq2, self.__heightSamplingFreq, order)
            #w, h = signal.freqz(b, a, worN=5000)
            w, h = signal.sosfreqz(sos, worN=5000)
            # TODO: [3]
            
            # Filter a noisy signal.
            uButter = []
            vButter = []
            tempButter = []
            for i,element in enumerate(vPert):
                
                filtU = self.__butter_bandpass_filter(uPert[i],freq1, freq2, self.__heightSamplingFreq, order)
                uButter.append(filtU)
                filtV = self.__butter_bandpass_filter(vPert[i], freq1, freq2, 1/5, order)
                vButter.append(filtV)
                filtTemp = self.__butter_bandpass_filter(tempPert[i],freq1, freq2, self.__heightSamplingFreq, order)
                tempButter.append(filtTemp)
                #axs[1,1].plot(vPert[0], Alt.magnitude)
                axs[1,1].plot(vButter[i], Alt.magnitude/1000, linewidth=0.5)
                axs[0,1].plot(uButter[i], Alt.magnitude/1000, linewidth=0.5)
                #plt.xlabel('time (seconds)')
                #plt.hlines([-a, a], 0, T, linestyles='--')
                #plt.grid(True)
                #plt.axis('tight')
                #plt.legend(loc='upper left')
        
            #re define u,v - these are the values used in analysis
            u = self.__butter_bandpass_filter(u.magnitude,freq1, freq2, self.__heightSamplingFreq, order)
            v = self.__butter_bandpass_filter(v.magnitude,freq1, freq2, self.__heightSamplingFreq, order)
            Temp = self.__butter_bandpass_filter(Temp.magnitude,freq1, freq2, self.__heightSamplingFreq, order)
            print("Butterworth Filter Applied")
        #polyIndice = backgroundPolyOrder - 2
        #u = uPert[polyIndice] * units.m / units.second
        #v = vPert[polyIndice] * units.m / units.second
        #Temp = tempPert[polyIndice] * units.degC
        #remove units
        Alt = Alt.magnitude
        #plot raw/filtered wind profiles
        # TODO: [2]
        #fig.savefig('pertCamparison.tiff', dpi=600)#, format="tiff")
        processedData = []
        return processedData
    

    def __butter_bandpass(lowcut, highcut, fs, order):
        """
            Filter design, also used for plotting the frequency response of Butterworth
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        #b, a = signal.butter(order, [low, high], btype='bandpass')
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        #return b, a
        return sos

    def preprocessDataResample(self, filepath):
        """ prepare data for hodograph analysis. non numeric values & values > 999999 removed, brunt-viasala freq
            calculated, background wind removed
            Different background removal techniques used: rolling average, savitsky-golay filter, nth order polynomial fits
        """
        # Open file
        contents = "" 
        f = open(filepath, 'r')
        #print("\nOpening file "+file+":")
        for line in f:
            if line.rstrip() == "Profile Data:":
                contents = f.read()  # Read in rest of file, discarding header
                #print("File contains GRAWMET profile data")
                break
        f.close()  # Need to close opened file
        # Read in the data and perform cleaning
        # Need to remove space so Virt. Temp reads as one column, not two
        contents = contents.replace("Virt. Temp", "Virt.Temp")
        # Break file apart into separate lines
        contents = contents.split("\n")
        contents.pop(1)  # Remove units so that we can read table
        index = -1  # Used to look for footer
        for i in range(0, len(contents)):  # Iterate through lines
            if contents[i].strip() == "Tropopauses:":
                index = i  # Record start of footer
                tropopauses = [int(s) for s in contents[i+1].split() if s.isdigit()]
                tropopauseMin = min(tropopauses)
                print("Highest Troopause: ", tropopauseMin)
        if index >= 0:  # Remove footer, if found
            contents = contents[:index]
        contents = "\n".join(contents)  # Reassemble string
        # format flight data in dataframe
        df = pd.read_csv(StringIO(contents), delim_whitespace=True)
        #turn strings into numeric data types, non numerics turned to nans
        df = df.apply(pd.to_numeric, errors='coerce') 
        # replace all numbers greater than 999999 with nans
        df = df.where(df < 999999, np.nan)    
        #truncate data at greatest alt
        df = df[0 : np.where(df['Alt']== df['Alt'].max())[0][0]+1]  
        #Truncate data below tropopause
        df = df[df['P'] <= tropopauseMin] 
        #drop rows with nans
        df = df.dropna(subset=['Time', 'T', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt'])
        #remove unneeded columns
        df = df[['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.']]

        #individual series for each variable, local
        rawData = {
            'Time':df['Time'].to_numpy(),
            'Pres':df['P'].to_numpy() * units.hPa,
            'Temp' : df['T'].to_numpy()  * units.degC,
            'Ws' : df['Ws'].to_numpy() * units.m / units.second,
            'Wd' : df['Wd'].to_numpy() * units.degree,
            'Long' : df['Long.'].to_numpy(),
            'Lat' : df['Lat.'].to_numpy(),
            'Alt' : df['Alt'].to_numpy().astype(int) * units.meter,
            'bv2' : None,
            'u' : None,
            'v' : None
            }
        #convert wind from polar to cartesian c.s.
        rawData['u'], rawData['v'] = mpcalc.wind_components(rawData['Ws'], rawData['Wd'])   #raw u,v components - no different than using trig fuctions
        
        return (rawData,df)