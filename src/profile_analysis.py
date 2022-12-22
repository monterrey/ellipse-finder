from src import constants
from src import params
from src import wavelet_analysis
from src import hodograph_analysis
class ProfileAlanyser:
    def __init__(self) -> None:
        print("in func")
        pass
    def hodoAnalysis():
        pass
    def waveletAnalysis(self):
        self.wave = wavelet_analysis.WaveletAnalyser(params.applyButterworth,constants.P_0, constants.HEIGHT_SAMPLING_FREQ,params.fileToBeInspected)

    def batchAnalysis(self, hodoBatch :bool , waveletBatch :bool ):
        pass
