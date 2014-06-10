#!/usr/bin/env python

import sys, argparse

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from drive_test_analysis import trace_loader as tl
from drive_test_analysis import nemo_trace_processor as ntp
from drive_test_analysis import data_plotter as dpl

def setup_args():
    parser = argparse.ArgumentParser(description='Plot drive test data.')
    parser.add_argument('-l','--list', action='store_true', help='List all data-set.')
    parser.add_argument('-s','--static', action='store_true', help='Keep samples with zero velocity.')
    parser.add_argument('-d','--select', type=int,
                        help='Select a particular data-set to display')
    parser.add_argument('--print', type=str,help='Print figure to file.')
    parser.add_argument('library',type=str, nargs='+',
                        help='Select a particular library to pull data from')
    args   = parser.parse_args()
    return args


def main(args):
    data_file_list = tl.get_data_file_list(args.library)
    if args.list:
        tl.print_list(data_file_list)
        sys.exit(0)
    data = tl.load_data_file(data_file_list,args.select)

    if not args.static:
        logging.debug('Remove zero velocity samples')
        data = ntp.remove_non_positive_velocity_samples(data)

    # Get basic data
    ntp.process_data(data,ntp.process_velocity)
    ntp.process_data(data,ntp.process_lte_bw)
    ntp.process_data(data,ntp.process_lte_prb_util)
    ntp.process_data(data,ntp.process_lte_prb_util_bw)

    column_list = ['Velocity', 'Velocity full',
                   'DL bandwidth','DL bandwidth full',
                   'PRB utilization DL','PRB utilization DL full',
                   'PRB utilization DL 10','PRB utilization DL 15']
    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data


    print(df['PRB utilization DL'].describe())
    print(df['PRB utilization DL 10'].describe())
    print(df['PRB utilization DL 15'].describe())
    plt.ion()
    plt.figure()
    x = np.arange(0,100,1)
    dpl.plot_ecdf_triplet(df['PRB utilization DL'].dropna(),
                          df['PRB utilization DL 10'].dropna(),
                          df['PRB utilization DL 15'].dropna(),x,
                          'PRB util.','PRB util. 10 MHz','PRB util. 15 MHz','%')
    plt.legend (loc=0,prop={'size':14})
    if args.print:
        plt.savefig(args.print,dpi=300,bbox_inches='tight')

    input('Press any key')
    

if __name__ == "__main__":
    args = setup_args()
    main(args)
