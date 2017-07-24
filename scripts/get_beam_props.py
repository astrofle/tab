#!/usr/bin/env python

import re
import glob
import h5py
import pickle
import logging

import numpy as np

from tab import tools as tabt

def main(filenames, output, fields=['name', 'ra', 'dec', 'mjd_start', 'mjd_end', 'freq', 'stations']):
    """
    """
    
    logger = logging.getLogger(__name__)

    logger.info('Creating a dictionary with keys: {0}'.format(fields))
    binfo = dict.fromkeys(fields)

    for i,field in enumerate(binfo.keys()):
        if field in ['name', 'mjd_start', 'mjd_end']:
            binfo[field] = np.zeros(len(filenames), dtype="S64")
        elif field == 'stations':
            binfo[field] = np.zeros(len(filenames), dtype=object)
        else:
            binfo[field] = np.zeros(len(filenames), dtype=np.float64)

    for i,filename in enumerate(filenames):

        logger.info('Working on file: {0}'.format(filename))
        
        f = h5py.File(filename, 'r')

        props = tabt.get_props_from_filename(filename)

        head1 = f['/'].attrs.items()
        head2 = f['/SUB_ARRAY_POINTING_{0}'.format(props['SAP'])].attrs.items()
        head3 = f['/SUB_ARRAY_POINTING_{0}/BEAM_{1}'.format(props['SAP'], props['B'])].attrs.items()
        axis2 = f['/SUB_ARRAY_POINTING_{0}/BEAM_{1}/COORDINATES/COORDINATE_1'.format(props['SAP'], props['B'])].attrs.items()[1][1]

        binfo['name'][i] = 'SAP{0}_B{1}'.format(props['SAP'], props['B'])
        binfo['ra'][i] = head3[17][1]
        binfo['dec'][i] = head3[20][1]
        binfo['mjd_start'][i] = head1[-2][1]
        binfo['mjd_end'][i] = head1[9][1]
        binfo['freq'][i] = np.mean(axis2)
        binfo['stations'][i] = head1[11][1]
        #binfo['mean'][i] = np.mean(data[:,fidx[0]:fidx[1]])

    # Save the dictionary with the beam information
    logger.info('Saving beam information in: {0}'.format(output))
    with open(output, 'wb') as out:
        pickle.dump(binfo, out, protocol=pickle.HIGHEST_PROTOCOL)

def parse_args():
    """
    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('filenames', type=str,
                        help='Files with the data.\n')
    parser.add_argument('output', type=str,
                        help='File where to save the processed data.\n')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=None, level=logging.DEBUG, format=formatter)
    logger = logging.getLogger(__name__)

    main(args.filenames, args.output)
