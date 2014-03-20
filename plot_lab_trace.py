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
    parser.add_argument('-d','--select', type=int,
                        help='Select a particular data-set to display')
    parser.add_argument('library',type=str,
                        help='Select library to pull data from')
    parser.add_argument('-u','--ue',type=str, default='e398',
                        help='What UE was used [e398|e3276]. Default is e398 ')
    parser.add_argument('--print', type=str,nargs='+',help='Print figure to file.')
    args   = parser.parse_args()
    return args

def main(args):
    data_file_list = tl.get_data_file_list(args.library)
    if args.list:
        tl.print_list(data_file_list)
        sys.exit(0)
    data = tl.load_data_file(data_file_list,args.select)

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
    ntp.process_data(data,ntp.process_lte_se)

    column_list = ['DL bandwidth','DL bandwidth full',
                   'Application throughput downlink',
                   'PRB utilization DL','PRB utilization DL full',
                   'SE norm','SE RB norm']
    if args.select is None:
        df = dl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data

    print(df['SE RB norm'].describe())
    print(df['SE norm'].describe())
    print(df['Application throughput downlink'].describe())

    # Normalize
    f_norm = lambda x: x/1e6

    plt.ion()
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    dpl.plot_ts(df['Application throughput downlink'].dropna(),
                'MAC DL throughput' if args.ue == 'e398' else 'App. throughput',
                'Mbit/s',marker_size=10,ylim=None)
    plt.subplot2grid((2,1), (1,0))
    x = np.arange(0,155,1)
    dpl.plot_ecdf(df['Application throughput downlink'].dropna().apply(f_norm),x,
                  'App. throughput','Mbit/s')

    plt.figure()
    dpl.plot_ecdf_pair(df['SE norm'].dropna(),
                   df['SE RB norm'].dropna(),np.linspace(0,8.5,86),
                   'SE norm','SE RB norm',
                   'bit/s/Hz')

    input('Press any key')
    

if __name__ == "__main__":
    args = setup_args()
    main(args)
