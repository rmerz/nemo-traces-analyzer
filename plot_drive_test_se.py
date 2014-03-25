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

    column_list = ['DL bandwidth','DL bandwidth full',
                   'PRB utilization DL','PRB utilization DL full',
                   'PRB utilization DL 10','PRB utilization DL 15','PRB utilization DL 20',
                   'SE','SE norm','SE 10 norm','SE 15 norm','SE 20 norm']
    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data


    print(df['SE'].describe())
    print(df['SE norm'].describe())
    print(df['SE 10 norm'].describe())
    print(df['SE 15 norm'].describe())
    print(df['SE 20 norm'].describe())
    plt.ion()

    plt.figure()
    plt.subplot2grid((2,2), (0,0),colspan=2)
    x = np.arange(0,8.1,0.1)
    dpl.plot_ecdf_pair(df['SE norm'].dropna(),
                       df['SE'].dropna(),x,
                       'Spectral efficiency (PRB norm.)',
                       'Spectral efficiency',
                       'bit/s/Hz')
    plt.legend(loc=0,prop={'size':10})
    plt.subplot2grid((2,2), (1,0),colspan=1)
    dpl.plot_density(df['SE'].replace([np.inf, -np.inf], np.nan).dropna(),x,
                     'Spectral efficiency','bit/s/Hz')
    plt.subplot2grid((2,2), (1,1),colspan=1)
    dpl.plot_density(df['SE norm'].where(df['SE norm'] < 8).replace([np.inf, -np.inf], np.nan).dropna(),x,
                     'Spectral efficiency (PRB norm.)','bit/s/Hz')

    if args.print:
        plt.savefig(args.print[0],dpi=300,bbox_inches='tight')

    plt.figure()
    if len(df['SE 15 norm'].dropna()) > 0:
        dpl.plot_ecdf_triplet(df['SE 10 norm'].dropna(),
                              df['SE norm'].dropna(),
                              df['SE 15 norm'].dropna(),x,
                              'Spectral efficiency 10 MHz (PRB norm.)\n',
                              'Spectral efficiency (PRB norm.)\n',
                              'Spectral efficiency 15 MHz (PRB norm.)',
                              'bit/s/Hz')
    else:
        dpl.plot_ecdf_pair(df['SE 10 norm'].dropna(),
                           df['SE norm'].dropna(),x,
                           'Spectral efficiency 10 MHz (PRB norm.)\n',
                           'Spectral efficiency (PRB norm.)\n',
                           'bit/s/Hz')
    plt.legend(loc=0,prop={'size':10})
    if args.print:
        plt.savefig(args.print[1],dpi=300,bbox_inches='tight')

    input('Press any key')
    

if __name__ == "__main__":
    args = setup_args()
    main(args)
