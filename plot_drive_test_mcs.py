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
    parser.add_argument('library',type=str,
                        help='Select a particular library to pull data from')
    parser.add_argument('--rank', type=int, help='Look at only a particular rank.')
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
    if args.rank is None:
        ntp.process_data(data,ntp.process_pdsch_mcs)
    elif args.rank == 1:
        ntp.process_data(data,ntp.process_pdsch_mcs_rank_1)
    elif args.rank == 2:
        ntp.process_data(data,ntp.process_pdsch_mcs_rank_2)
    else:
        assert("You should not be here: rank must be equal to 1 or 2.")

    column_list = ['Velocity', 'Velocity full',
                   'valid_percentage',
                   'mcs_q_2','mcs_q_4','mcs_q_6','mcs_reserved','mcs_na']

    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data[column_list]

    mcs_index = df['valid_percentage'].dropna().index.values

    print(df[['mcs_q_2']].dropna().describe())
    print(df[['mcs_q_4']].dropna().describe())
    print(df[['mcs_q_6']].dropna().describe())
    print(df[['mcs_reserved']].dropna().describe())
    print(df[['mcs_na']].dropna().describe())
    print(df[['mcs_q_2','mcs_q_4','mcs_q_6','mcs_reserved','mcs_na']].sum(axis=1).dropna().median())
    plt.ion()

    plt.figure()
    plt.subplot2grid((1,5), (0,0),colspan=1)
    plt.boxplot(df[['mcs_q_2']].dropna().values)
    plt.ylim([0,100])
    plt.grid(True)
    plt.xticks([])
    plt.title('MCS 0-9,\nQ_m = 2',fontsize=10)

    plt.subplot2grid((1,5), (0,1),colspan=1)
    plt.boxplot(df[['mcs_q_4']].dropna().values)
    plt.ylim([0,100])
    plt.grid(True)
    plt.xticks([])
    plt.title('MCS 10-16,\nQ_m = 4',fontsize=10)

    plt.subplot2grid((1,5), (0,2),colspan=1)
    plt.boxplot(df[['mcs_q_6']].dropna().values)
    plt.ylim([0,100])
    plt.grid(True)
    plt.xticks([])
    plt.title('MCS 17-28,\nQ_m = 6',fontsize=10)

    plt.subplot2grid((1,5), (0,3),colspan=1)
    plt.boxplot(df[['mcs_reserved']].dropna().values)
    plt.ylim([0,100])
    plt.grid(True)
    plt.xticks([])
    plt.title('MCS 29-31,\nreserved',fontsize=10)

    plt.subplot2grid((1,5), (0,4),colspan=1)
    plt.boxplot(df[['mcs_na']].dropna().values)
    plt.ylim([0,100])
    plt.grid(True)
    plt.xticks([])
    plt.title('MCS n/a',fontsize=10)

    plt.tight_layout()

    if args.print:
        plt.savefig(args.print,dpi=300,bbox_inches='tight')

    input('press any key')

if __name__ == "__main__":
    args = setup_args()
    main(args)
