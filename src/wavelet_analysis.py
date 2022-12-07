from flight_profile import FlightProfiler
from metpy.units import units
class WaveletAnalyser(FlightProfiler):
    def __init__(self, applyButterworth:bool, p_0: units.hPa , heightSamplingFreq : units.m ) -> None:
        super().__init__(applyButterworth, p_0, heightSamplingFreq  )
