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
    parser.add_argument('--print', type=str,help='Print figure to file.')  # , nargs='+'
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
    ntp.process_data(data,ntp.process_lte_bw)

    column_list = ['DL bandwidth','DL bandwidth full']
    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data

    print(df['DL bandwidth'].describe())
    print('Percentage of each bandwidth:')
    print(df['DL bandwidth'].dropna().value_counts(normalize=True))
    bandwidth_transitions_count = df['DL bandwidth'].dropna().diff().value_counts(normalize=True)
    print('Percentage of bandwidth transitions: ',bandwidth_transitions_count[bandwidth_transitions_count.index != 0].sum())

    plt.ion()
    plt.figure()
    plt.subplot2grid((1,5),(0,0),colspan=4)
    dpl.plot_ts(df['DL bandwidth'].dropna(),
                'DL bandwidth','MHz',
                10,[9.5,15.5])
    plt.subplot2grid((1,5),(0,4),colspan=1)

    bandwidth_counts = df['DL bandwidth'].dropna().value_counts(normalize=True)
    per10 = bandwidth_counts[bandwidth_counts.index == 10]*100
    per15 = bandwidth_counts[bandwidth_counts.index == 15]*100

    # print(per10,per15)
    plt.bar(1,per15,width=0.4,bottom=per10,color='Orange',align='center',
            label='15 MHz')
    plt.bar(1,per10,width=0.4,color='m',align='center',
            label='10 MHz')
    plt.grid(True)
    plt.legend()
    plt.xlim([0,2])
    plt.xticks([])
    plt.tight_layout()
    if args.print:
        plt.savefig(args.print,dpi=300,bbox_inches='tight')

    input('Press any key')
    

if __name__ == "__main__":
    args = setup_args()
    main(args)
