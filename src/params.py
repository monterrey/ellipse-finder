from metpy.units import units
from pathlib import Path
import os
showVisualizations = True    # Displays macroscopic hodograph for flight
siftThruHodo = False   # Use manual GUI to locate ellipse-like structures in hodograph
analyze = True   # Display list of microhodographs with overlayed fit ellipses as well as wave parameters
applyButterworth = True #should Butterworth filter be applied to data? Linear interpolation is also implemented, prior to filtering, at specified spatial resolution
location = "Tolten"     #[Tolten]/[Villarica]

backgroundPolyOrder = 3
applyButterworth = True

dir = Path()
cwd = dir.resolve().parent
flightData = dir.joinpath('test-data/Tolten/')             #flight data directory
fileToBeInspected = flightData.joinpath('T26_1630_12142020_MT2.txt')                                                                       #specific flight profile to be searched through manually
microHodoDir = cwd.joinpath('test-data/Tolten/test/') #"Tolten_butterNoSubtraction/T29"
waveParamDir = cwd.joinpath('test-data/Tolten/waveParams')   #location where wave parameter files are to be saved
microHodoFolderDir = cwd.joinpath('test-data/Tolten/T28')    #location of micrhodo folders for each flight
#configuration file
configFile = "Tolten_FlightTimes.csv"
configPath = cwd.joinpath('test-data/Tolten')

if location == "Tolten":
    latitudeOfAnalysis = abs(-39.236248) * units.degree    #latitude at which sonde was launched. Used to account for affect of coriolis force.
elif location == "Villarica":
    latitudeOfAnalysis = abs(-39.30697) * units.degree     #same, but for Villarica...