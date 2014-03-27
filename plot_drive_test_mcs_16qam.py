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
    parser.add_argument('library',type=str, nargs='+',
                        help='Select a particular library to pull data from')
    parser.add_argument('--print', type=str, help='Print figure to file.')
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

    ntp.process_data(data,ntp.process_velocity)
    ntp.process_data(data,ntp.process_pdsch_mcs_16qam)

    column_list = ['mcs_10','mcs_11','mcs_12','mcs_13','mcs_14','mcs_15','mcs_16']

    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data[column_list]

    plt.ion()
    plt.figure()

    for index in np.arange(0,7):
        print(index,index+10)
        print(df[['mcs_'+str(index+10)]].dropna().describe())
        plt.boxplot(df[['mcs_'+str(index+10)]].dropna().values,positions=[index+1])
    plt.ylim([0,100])
    plt.xlim([0,8])
    plt.xticks(np.arange(1,8),['MCS '+str(index+10) for index in np.arange(0,7)],rotation=45)
    plt.grid(True)
    plt.tight_layout()

    if args.print:
        plt.savefig(args.print,dpi=300,bbox_inches='tight')

    input('press any key')

if __name__ == "__main__":
    args = setup_args()
    main(args)
