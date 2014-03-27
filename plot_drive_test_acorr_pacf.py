#!/usr/bin/env python

import sys, argparse

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

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

    # Spectral efficiency, SNR, RSRP
    ntp.process_data(data,ntp.process_se_bw_norm)
    ntp.process_data(data,ntp.process_lte_rs_snr)
    ntp.process_data(data,ntp.process_lte_rsrp)

    column_list = ['Velocity',
                   'SE','SE norm',
                   'RS SNR/Antenna port - 1','RS SNR/Antenna port - 2',
                   'RSRP/Antenna port - 1','RSRP/Antenna port - 2']
    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data

    # Remove outliers because of bandwidth normalization issues
    df['SE norm'][df['SE norm'] > 7.5] = np.nan

    print(df['SE'].dropna().describe())
    print(df['Velocity'].dropna().describe())
    print(df['RS SNR/Antenna port - 1'].dropna().describe())

    velocity_pacf = pacf(df['Velocity'].dropna(), nlags=10, method='ywunbiased', alpha=None)
    se_pacf,se_conf = pacf(df['SE'].dropna(), nlags=10, method='ywunbiased', alpha=0.05)
    se_norm_pacf,se_norm_conf = pacf(df['SE norm'].dropna(), nlags=10, method='ywunbiased', alpha=0.05)
    rs_snr_ap1_pacf = pacf(df['RS SNR/Antenna port - 1'].dropna(), nlags=10, method='ywunbiased', alpha=None)
    rsrp_ap1_pacf = pacf(df['RSRP/Antenna port - 1'].dropna(), nlags=10, method='ywunbiased', alpha=None)

    # Apply diff to ensure zero-mean
    velocity_acf = acf(df['Velocity'].dropna().diff().dropna(), unbiased=False, nlags=40, confint=None, qstat=False, fft=True, alpha=None)
    se_acf = acf(df['SE'].dropna().diff().dropna(), unbiased=False, nlags=40, confint=None, qstat=False, fft=True, alpha=None)
    se_norm_acf = acf(df['SE norm'].dropna().diff().dropna(), unbiased=False, nlags=40, confint=None, qstat=False, fft=True, alpha=None)
    rs_snr_ap1_acf = acf(df['RS SNR/Antenna port - 1'].dropna().diff().dropna(), unbiased=False, nlags=40, confint=None, qstat=False, fft=True, alpha=None)
    rsrp_ap1_acf = acf(df['RSRP/Antenna port - 1'].dropna().diff().dropna(), unbiased=False, nlags=40, confint=None, qstat=False, fft=True, alpha=None)

    plt.ion()
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    plt.plot(se_pacf,lw=2.0,label='PACF SE')
    plt.plot(se_norm_pacf,lw=2.0,label='PACF SE norm')
    plt.plot(rs_snr_ap1_pacf,lw=2.0,label='PACF RS SNR AP1')
    plt.plot(rsrp_ap1_pacf,lw=2.0,label='PACF RSRP AP1')
    plt.plot(velocity_pacf,lw=2.0,label='PACF Velocity')
    plt.ylim([-0.2,1.1])
    plt.grid(True)
    plt.legend(loc=0)

    plt.subplot2grid((2,1), (1,0))
    plt.plot(se_acf,lw=2.0,label='ACF SE diff')
    plt.plot(se_norm_acf,lw=2.0,label='ACF SE norm diff')
    plt.plot(rs_snr_ap1_acf,lw=2.0,label='ACF RS SNR AP1 diff')
    plt.plot(rsrp_ap1_acf,lw=2.0,label='ACF RSRP AP1 diff')
    plt.plot(velocity_acf,lw=2.0,label='ACF Velocity diff')
    plt.ylim([-0.2,1.1])
    plt.grid(True)
    plt.legend(loc=0)

    plt.tight_layout()

    input('Press any key')
    

if __name__ == "__main__":
    args = setup_args()
    main(args)
