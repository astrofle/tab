#!/usr/bin/env python

import re
import h5py 
import logging

import numpy as np

from astropy.stats import sigma_clip

logger = logging.getLogger(__name__)

def flag_subband_edges(freq, data, ech=10):
    """
    """
    
    data.mask[:,:,:ech] = True
    data.mask[:,:,-ech:] = True
    freq.mask[:,:ech] = True
    freq.mask[:,-ech:] = True
    
    return freq, data

def get_h5_data(filename):
    """
    """
    
    props = get_props_from_filename(filename)
    data_table = 'SUB_ARRAY_POINTING_{0}/BEAM_{1}/STOKES_{2}'.format(props['SAP'], props['B'], props['S'])
    freq_table = '/SUB_ARRAY_POINTING_{0}/BEAM_{1}/COORDINATES/COORDINATE_1'.format(props['SAP'], props['B'])

    f = h5py.File(filename)
    data = np.ma.masked_invalid(f[data_table])
    freq = f[freq_table].attrs.items()[1][1]
    
    head3 = f['/SUB_ARRAY_POINTING_{0}/BEAM_{1}'.format(props['SAP'], props['B'])].attrs.items()
    
    nchan = head3[-2][1]/head3[9][1]
    
    freq = freq.reshape((data.shape[1]/nchan, nchan))
    data = data.reshape((data.shape[0], data.shape[1]/nchan, nchan))
    
    return freq, data

def get_props_from_filename(filename, props=['SAP', 'S', 'B']):
    """
    """

    fnprops = dict.fromkeys(props)

    for i,prop in enumerate(props):
        fnprops[prop] = re.findall('{0}\d+'.format(prop), filename)[-1][len(prop):]

    return fnprops

def sigma_clip_per_subband(freq, data):
    """
    """
    
    mdata = np.ma.masked_invalid(data, copy=True)
    mfreq = np.ma.masked_invalid(freq, copy=True)
    
    for i in xrange(data.shape[0]):
        s = data[i]
        mdata[i] = sigma_clip(s)
        mfreq[i] = np.ma.masked_where(mdata[i].mask, freq[i])
        
    return mfreq, mdata
