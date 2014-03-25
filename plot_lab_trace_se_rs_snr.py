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
    parser.add_argument('library',type=str, nargs='+',
                        help='Select a particular library to pull data from')
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
    # RSRP and RS-SNR
    ntp.process_data(data,ntp.process_lte_rsrp_rs_snr)
    ntp.process_data(data,ntp.process_lte_rsrp_rs_snr_full)
    # Spectral efficiency
    ntp.process_data(data,ntp.process_lte_se_rb)

    column_list = ['DL bandwidth','DL bandwidth full',
                   'Application throughput downlink',
                   'PRB utilization DL','PRB utilization DL full',
                   'RSRP/Antenna port - 1 full', 'RSRP/Antenna port - 2 full',
                   'RS SNR/Antenna port - 1 full', 'RS SNR/Antenna port - 2 full',
                   'SE RB norm']
    if args.select is None:
        df = dl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data

    print(df['SE RB norm'].describe())
    print(df['Application throughput downlink'].describe())
    print(df['RS SNR/Antenna port - 1 full'].describe())
    print(df['RS SNR/Antenna port - 2 full'].describe())
    print(df['RSRP/Antenna port - 1 full'].describe())
    print(df['RSRP/Antenna port - 2 full'].describe())

    # Normalize
    f_norm = lambda x: x/1e6

    plt.ion()
    index = df[['SE RB norm','RS SNR/Antenna port - 1 full']].dropna().index
    plt.subplot2grid((1,2), (0,0))
    dpl.plot_scatter(df['RS SNR/Antenna port - 1 full'][index],df['SE RB norm'][index],
                     'RS SNR AP1','SE RB',
                     'dB','bit/s/Hz',marker_size=20,alpha=.3)
    plt.subplot2grid((1,2), (0,1))
    dpl.plot_scatter(df['RS SNR/Antenna port - 2 full'][index],df['SE RB norm'][index],
                     'RS SNR AP2','SE RB',
                     'dB','bit/s/Hz',marker_size=20,alpha=.3)

    plt.figure()
    plt.subplot2grid((1,2), (0,0))
    dpl.plot_scatter(df['RSRP/Antenna port - 1 full'][index],df['SE RB norm'][index],
                     'RSRP AP1','SE RB',
                     'dBm','bit/s/Hz',marker_size=20,alpha=.3)
    plt.subplot2grid((1,2), (0,1))
    dpl.plot_scatter(df['RSRP/Antenna port - 2 full'][index],df['SE RB norm'][index],
                     'RSRP AP2','SE RB',
                     'dBm','bit/s/Hz',marker_size=20,alpha=.3)

    input('Press any key')
    

if __name__ == "__main__":
    args = setup_args()
    main(args)

