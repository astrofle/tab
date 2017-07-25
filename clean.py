#!/usr/bin/env python

import numpy as np
import scipy.optimize as opt
from scipy.interpolate import griddata

from astropy import units as u
from astropy.coordinates import Angle

from . import functions

from lofholog import coordinates

def clean_image(dirty_map, psf, mapcoo, psfcoo, loop_gain=0.1, iterations=1e3, interp_order=3, mask=''):
    """
    Clean a 2D complex valued image.
    TODO: implements masking
    """
    
    psf_x = psfcoo[0]
    psf_y = psfcoo[1]
    psf_xx, psf_yy = np.meshgrid(psfcoo[0], psfcoo[1])
    
    map_xx, map_yy = mapcoo[0], mapcoo[1]

    norm_psf = np.ma.masked_invalid(psf/psf.max())
    
    residuals = dirty_map.copy()
    
    rm_points = np.zeros((int(iterations)), dtype=int)
    rm_points_amp = np.zeros((int(iterations)), dtype=np.complex128)
    clean_map = np.zeros(dirty_map.shape, dtype=dirty_map.dtype)
    
    for i in xrange(int(iterations)):
        
        dm_i = residuals.ravel()
        dm_max = np.ma.max(abs(dm_i))
        idx = np.where(abs(dm_i) == dm_max)[0][-1]
        #idx = np.argmax(abs(dm_i))
        #if idx != np.argmax(abs(dm_i)):
            #print i
        #print len(np.where(abs(dm_i) == dm_max)[0])
        #print idx, np.argmax(abs(dm_i))

        rm_points[i] = idx
        x_offset = mapcoo[0][idx]
        y_offset = mapcoo[1][idx]

        psf_ra, psf_dec = coordinates.radec_from_lm(psf_x, psf_y, 
                                                    Angle(x_offset*u.deg), 
                                                    Angle(y_offset*u.deg))

        sampled_re = griddata(np.array([psf_ra.deg, psf_dec.deg]).T, norm_psf.real, 
                                  (mapcoo[0].reshape(dirty_map.shape),
                                   mapcoo[1].reshape(dirty_map.shape)), 
                              method='linear', 
                              fill_value=0, rescale=False)
        sampled_re = np.ma.masked_invalid(sampled_re)
        residuals = residuals - sampled_re*loop_gain*dm_i[idx]
        rm_points_amp[i] = loop_gain*dm_i[idx]

    # Fit a Gaussian to the PSF
    initial_guess = [0.15]
    gauss = lambda x, sx: functions.gaussian2d(x, 1., psf_ra[0].deg, sx, psf_dec[0].deg, sigma_y=False, rota=False).ravel()
    popt, pcov = opt.curve_fit(gauss, (psf_ra.deg, psf_dec.deg), abs(norm_psf), p0=initial_guess)
    sigma_x = popt[0]
    
    initial_guess = [0, sigma_x, 0, sigma_x, 45]
    gauss = lambda x, x0, sx, y0, sy, ra: functions.gaussian2d(x, 1., x0, sx, y0, sy, ra).ravel()
    popt, pcov = opt.curve_fit(gauss, (psf_ra, psf_dec), abs(norm_psf), p0=initial_guess)
    sigma_x = popt[1]
    sigma_y = popt[3]
    rot_ang = popt[4]
    
    for k in range(len(rm_points_amp)):
        clean_beam = functions.gaussian2d((mapcoo[0].reshape(dirty_map.shape), 
                                           mapcoo[1].reshape(dirty_map.shape)), 1., 
                                          center_x=mapcoo[0][rm_points[k]],
                                          center_y=mapcoo[1][rm_points[k]], 
                                          sigma_x=sigma_x, sigma_y=sigma_y, 
                                          rota=rot_ang)
        norm_clean_beam = (clean_beam/clean_beam.max())*rm_points_amp[k]
        clean_map = clean_map + norm_clean_beam
    
    return {'residuals':residuals,
            'CM':clean_map,
            'rm_points':rm_points,
            'rm_points_amp':rm_points_amp,
            'fwhmx_m':2.*np.sqrt(2.*np.log(2.))*sigma_x,
            'fwhmy_m':2.*np.sqrt(2.*np.log(2.))*sigma_y,
            'fwhmx_p':2.*np.sqrt(2.*np.log(2.))*sigma_x,
            'fwhmy_p':2.*np.sqrt(2.*np.log(2.))*sigma_y,
            'rota':rot_ang,
            'PSF':psf}