#!/usr/bin/env python

import logging
import argparse
import numpy as np

from datetime import datetime

from astropy.stats import sigma_clip

from tab import tools as tabt

def main(filename, output, ech=25):
    """
    """
    
    logger = logging.getLogger(__name__)

    # Load the data
    logger.info('Loading file: {0}'.format(filename))
    freq, data = tabt.get_h5_data(filename)

    # Flag edge channels
    logger.info('Flagging {0} edge channels.'.format(ech))
    efreq, edata = tabt.flag_subband_edges(np.ma.masked_invalid(freq), data, ech=ech)

    # Sigma clip the data on a per subband basis
    logger.info('Sigma clipping each subband.')
    scfreq, scdata = tabt.sigma_clip_per_subband(efreq, edata.mean(axis=0))

    # Sigma clip the whole frequency range
    logger.info('Sigma clipping all subbands.')
    fcube = sigma_clip(scdata.flatten())
    ffreq = scfreq.flatten()

    # Reshape to keep the (time, subband, channel) structure
    # Time is probably gone by now
    final = fcube.reshape(scdata.shape)
    faxis = ffreq.reshape(scdata.shape)

    # Prepare to write
    out = np.ma.zeros(final.shape+(2,))
    out[:,:,0] = faxis
    out[:,:,1] = final
    out.fill_value = np.nan

    logger.info('Writting processed data to: {0}'.format(output))
    np.save(output, out.filled())

def parse_args():
    """
    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('filename', type=str,
                        help='File with the data.\n')
    parser.add_argument('output', type=str,
                        help='File where to save the processed data.\n')
    parser.add_argument('-e', '--edge_channels', type=int, default=25,
                        help='Number of edge channels to flag.\n')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    startTime = datetime.now()
    
    args = parse_args()

    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=None, level=logging.DEBUG, format=formatter)
    logger = logging.getLogger(__name__)

    main(args.filename, args.output, args.edge_channels)

    logger.info('Script run time: {0}'.format(datetime.now() - startTime))
