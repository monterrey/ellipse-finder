import constants
import params
import flight_profile

from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def uvVisualize(u, v, uBackground,Alt,vBackground):
    """ plot u, v, background wind vs. altitude
    """
    #housekeeping
    plt.figure("U & V vs Time", figsize=(10,10)) 
    plt.suptitle('Smoothed U & V Components', fontsize=16)

    #u vs alt
    plt.subplot(1,2,1)
    plt.plot((u.magnitude + uBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(uBackground.magnitude, Alt.magnitude, label='Background - S.G.')
    plt.plot(u.magnitude, Alt.magnitude, label='De-Trended - S.G.')

    #plt.plot(uBackgroundRolling, Alt.magnitude, label='Background - R.M.')
    #plt.plot(uRolling, Alt.magnitude, label='De-Trended - R.M.')

    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.title("U")

    #v vs alt
    plt.subplot(1,2,2)
    plt.plot((v.magnitude + vBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(vBackground.magnitude, Alt.magnitude, label='Background - S.G.')
    plt.plot(v.magnitude, Alt.magnitude, label='De-Trended - S.G.')

    #plt.plot(vBackgroundRolling, Alt.magnitude, label='Background - R.M.')
    #plt.plot(vRolling, Alt.magnitude, label='De-Trended - R.M.')

    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.legend(loc='upper right', fontsize=16)
    plt.title("V")
    return