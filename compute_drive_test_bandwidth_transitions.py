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
    parser.add_argument('-u','--ue',type=str, default='e398',
                        help='What UE was used [e398|e3276]. Default is e398 ')
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

    if args.ue == 'e398':
        # Rename MAC downlink throughput in Application downlink throughput if need be
        ntp.process_data(data,ntp.process_lte_rename_mac_to_app)

    # Get basic data
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw20)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw10)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw15)
    # Spectral efficiency
    ntp.process_data(data,ntp.process_se_bw_norm)

    column_list = ['DL bandwidth','DL bandwidth full','SE','SE norm']
    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data

    # Check that overall results make sense
    print(df[df['SE norm'] <= 8][['SE','SE norm','DL bandwidth full']].dropna().describe())

    print('Percentage of each bandwidth:')
    print(df[df['SE norm'] <= 8]['DL bandwidth full'].dropna().value_counts(normalize=True))

    bandwidth_transitions_count = df[df['SE norm'] <= 8]['DL bandwidth full'].dropna().diff().value_counts(normalize=True)
    print('Percentage of bandwidth transitions: ',bandwidth_transitions_count[bandwidth_transitions_count.index != 0].sum())
    

if __name__ == "__main__":
    args = setup_args()
    main(args)
