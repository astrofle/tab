#!/usr/bin/env python

import numpy as np

functions = ('gaussian', 'gaussian2d')

def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """
    Return a 1-dimensional Gaussian function.
    
    
    """
    
    return (amplitude/(np.sqrt(2.*np.pi)*sigma)) * exp(-np.power((1.0*x-center)/(sigma), 2.)/2.)

def gaussian2d(x, amplitude=1.0, center_x=0.0, sigma_x=1.0, center_y=0.0, sigma_y=1.0, rota=0.0):
    """
    Return a 2-dimensional Gaussian function.
    
    
    """
    
    if len(x) == 1:
        y = x
    else:
        (x, y) = x
        
    if not sigma_y:
        sigma_y = sigma_x
        
    if not center_y:
        center_y = center_x
    
    if rota:
        center_x = center_x*np.cos(np.deg2rad(rota)) - center_y*np.sin(np.deg2rad(rota))
        center_y = center_x*np.sin(np.deg2rad(rota)) + center_y*np.cos(np.deg2rad(rota))        
    
        x = x*np.cos(np.deg2rad(rota)) - y*np.sin(np.deg2rad(rota))
        y = x*np.sin(np.deg2rad(rota)) + y*np.cos(np.deg2rad(rota))
    
    norm = 2.*np.pi*sigma_x*sigma_y
    #exp_x = np.power((x - center_x)/(sigma_x), 2.)
    #exp_y = np.power((y - center_y)/(sigma_y), 2.)
    g = amplitude*np.exp(-(((center_x - x)/sigma_x)**2 + \
                                ((center_y - y)/sigma_y)**2)/2.)
    
    return g #(amplitude/norm)*np.exp(-(exp_x + exp_y)/2.)

def gaussian2d_as1d(x, **kwargs):
    """
    Return a 2-dimensional Gaussian function as a 1D array.
    """
    
    g = gaussian2d(x, **kwargs)
    
    return g.ravel()

if __name__ == '__main__':
    pass