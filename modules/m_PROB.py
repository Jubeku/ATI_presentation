# Module for calculating error between simulations and observations

import numpy as np


def calc_prob(windowed_st, channels, ydim, xdim, freqID):
  
    """
    This function calculates the energy ratios of station pairs from the given 
    seismic signals and compares them to simulated energy ratios from each 
    grid position, i.e. the potential source locations. 

    Input parameters
    windowed_st:  Stream of seismic traces in a certain time window
    channels:     List of station channels (components E, N, Z) used for the 
                  localization
    ydim:         Grid dimension in y-direction (North)
    xdim:         Grid dimension in x-direction (East)
    freqID:       Frequency band ID
    
    """
    
    ref = 'BON'
    dt = windowed_st[0].stats.delta
    
    # Pre-allocate array with probabilities
    probs_tw = np.zeros((ydim,xdim))
    
    # Energy ratios are calculated for each component in respect to the chosen
    # reference station
    for ch in channels:                
        if ch == 'Z':
            stas_name = ['BOR','DSO','SNE']
        else:
            stas_name = ['BOR','SNE']

        nRatios = len(stas_name)
        ratio_obs = np.zeros(nRatios)
        
        # Energy ratios from observed signal
        energy_curr_ref = np.trapz( 
                windowed_st.select(station=ref,channel='*'+ch)[0].data**2, 
                dx=dt )
        for idxSta, sta in enumerate(stas_name):
            energy_curr = np.trapz(
                    windowed_st.select(station=sta,channel='*'+ch)[0].data**2, 
                    dx=dt )
            ratio_obs[idxSta] = energy_curr/energy_curr_ref

        # Energy ratios from simulated signal
        if ch == 'Z':
            direc_comp = 'data/simu/vertZ/'
        elif ch == 'E':
            direc_comp = 'data/simu/horzE/'
        elif ch == 'N':
            direc_comp = 'data/simu/horzN/'
        path_simu_ref = direc_comp + ref + '/' 
        energy_simu_ref = np.loadtxt(path_simu_ref + freqID + '/energy_Fz.txt')
        energy_simu_ref = np.reshape(energy_simu_ref,(ydim,xdim))
        ratios_simu = np.zeros((ydim,xdim,nRatios)) 
        for idxSrc, sta in enumerate(stas_name):
            path_simu = direc_comp  + sta + '/' 
            energy_simu = np.loadtxt(path_simu + freqID + '/energy_Fz.txt')
            energy_simu = np.reshape(energy_simu,(ydim,xdim))
            ratios_simu[:,:,idxSrc] = energy_simu/energy_simu_ref
                    
        # Error between observed and simulated energy ratios
        for ii in range(ydim):
            for jj in range(xdim):
                for kk in range(nRatios):
                    probs_tw[ii,jj] += 1./nRatios * (
                        np.abs(np.log10(ratios_simu[ii,jj,kk]/ratio_obs[kk])))

    # Probability is defined as inverse of error, normalized by 
    # number of channels
    probs_tw = 1./(probs_tw/len(channels))

    return(probs_tw)
