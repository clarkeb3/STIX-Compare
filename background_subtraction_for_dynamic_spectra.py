# Written by Eoin Carley
# Use to background subtract dynamic spectra data

import numpy as np

def backsub(data):
    # Devide each spectrum by the spectrum with the minimum standard deviation.
    data = np.log10(data)
    data[np.where(np.isinf(data)==True)] = 0.0
    data_std = np.std(data, axis=0)
    min_std_index = np.where(data_std==np.min( data_std[np.nonzero(data_std)] ))[0][0]
    min_std_spec = data[:, min_std_index]
    nfreq = len(min_std_spec)
    data = np.divide(data, min_std_spec.reshape(nfreq, 1))
    #Alternative: Normalizing frequency channel responses using median of values.
    #for sb in np.arange(data.shape[0]):
    #       data[sb, :] = data[sb, :]/np.mean(data[sb, :])
    return data
