#!/usr/bin/env python

import sys, argparse

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from scipy.stats import probplot,spearmanr,kendalltau
import statsmodels.api as sm
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from sklearn.decomposition import PCA

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
    parser.add_argument('--print', nargs='+', help='Print plots.')
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

    ntp.process_data(data,ntp.process_lte_app_bw_prb_util)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw10)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw15)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw20)
    ntp.process_data(data,ntp.process_lte_rsrp_rs_snr)
    ntp.process_data(data,ntp.process_lte_rsrp_rs_snr_full)
    ntp.process_data(data,ntp.process_se_bw_norm)
    ntp.process_data(data,ntp.process_velocity)

    column_list = ['Velocity', 'Velocity full',
                   'DL bandwidth','DL bandwidth full',
                   'PRB utilization DL','PRB utilization DL full',
                   'Application throughput downlink',
                   'Application throughput downlink norm',
                   'Application throughput downlink 10 norm','Application throughput downlink 15 norm',
                   'RS SNR/Antenna port - 1','RS SNR/Antenna port - 2',
                   'RS SNR/Antenna port - 1 full','RS SNR/Antenna port - 2 full',
                   'RSRP/Antenna port - 1','RSRP/Antenna port - 2',
                   'RSRP/Antenna port - 1 full','RSRP/Antenna port - 2 full',
                   'SE norm']

    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data

    # Remove outliers because of bandwidth normalization issues
    df['SE norm'][df['SE norm'] > 7.5] = np.nan

    print(df['Application throughput downlink'].describe())
    print(df['Application throughput downlink 10 norm'].describe())
    print(df['Application throughput downlink 15 norm'].describe())

    plt.ion()

    plt.figure()
    X = df[['RS SNR/Antenna port - 1 full','SE norm']].dropna()
    dpl.plot_scatter(X['RS SNR/Antenna port - 1 full'],
                     X['SE norm'],
                     'RS-SNR AP 1','Spectral efficiency',
                     'dB','bit/s/Hz')

    plt.figure()
    dpl.plot_hist2d(X['RS SNR/Antenna port - 1 full'],
                    X['SE norm'],
                    'RS-SNR AP 1','Spectral efficiency',
                    'dB','bit/s/Hz',
                    bins=50)
    if args.print:
        plt.savefig(args.print[0],dpi=300,bbox_inches='tight')

    plt.figure()
    X = df[['RS SNR/Antenna port - 2 full','SE norm']].dropna()
    dpl.plot_scatter(X['RS SNR/Antenna port - 2 full'],
                     X['SE norm'],
                     'RS-SNR AP 2','Spectral efficiency',
                     'dB','bit/s/Hz')

    plt.figure()
    dpl.plot_hist2d(X['RS SNR/Antenna port - 2 full'],
                    X['SE norm'],
                    'RS-SNR AP 2','Spectral efficiency',
                    'dB','bit/s/Hz',
                    bins=50)

    plt.figure()
    X = df[['Velocity','SE norm']].dropna()
    dpl.plot_scatter(X['Velocity'],
                     X['SE norm'],
                     'Velocity','Spectral efficiency',
                     'km/h','bit/s/Hz')

    plt.figure()
    dpl.plot_hist2d(X['Velocity'],
                     X['SE norm'],
                    'Velocity','Spectral efficiency',
                     'km/h','bit/s/Hz',bins=50)
    if args.print:
        plt.savefig(args.print[1],dpi=300,bbox_inches='tight')

    plt.figure()
    X = df[['Velocity','RS SNR/Antenna port - 1']].dropna()
    dpl.plot_hist2d(X['Velocity'],
                    X['RS SNR/Antenna port - 1'],
                    'Velocity','RS-SNR AP 1',
                    'km/h','dB',bins=50)

    if args.print:
        plt.savefig(args.print[2],dpi=300,bbox_inches='tight')

    plt.figure()
    X = df[['Velocity','RS SNR/Antenna port - 2']].dropna()
    dpl.plot_hist2d(X['Velocity'],
                    X['RS SNR/Antenna port - 2'],
                    'Velocity','RS-SNR AP 2',
                    'km/h','dB',bins=50)

    input('Press any key.')

if __name__ == "__main__":
    args = setup_args()
    main(args)
