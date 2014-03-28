#!/usr/bin/env python

import sys, argparse

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from scipy.stats import spearmanr
from statsmodels.formula.api import glsar
from statsmodels.tsa.stattools import pacf
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
    parser.add_argument('--print', type=str, help='Print plots.')
    args   = parser.parse_args()
    return args

def main(args):
    data_file_list = tl.get_data_file_list(args.library)
    if args.list:
        tl.print_list(data_file_list)
        sys.exit(0)
    data = tl.load_data_file(data_file_list,args.select)

    if not args.static:
        logging.debug('Remove non-zero velocity samples')
        data = ntp.remove_non_positive_velocity_samples(data)

    if args.ue == 'e398':
        # Rename MAC downlink throughput in Application downlink throughput if need be
        ntp.process_data(data,ntp.process_lte_rename_mac_to_app)

    ntp.process_data(data,ntp.process_velocity)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw20)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw10)
    ntp.process_data(data,ntp.process_lte_app_bw_prb_util_bw15)
    ntp.process_data(data,ntp.process_se_bw_norm)
    ntp.process_data(data,ntp.process_lte_rsrp_rs_snr)
    ntp.process_data(data,ntp.process_lte_rsrp_rs_snr_full)

    column_list = ['Velocity', 'Velocity full',
                   'DL bandwidth','DL bandwidth full',
                   'PRB utilization DL','PRB utilization DL full',
                   'Application throughput downlink',
                   'Application throughput downlink norm',
                   'Application throughput downlink 10 norm','Application throughput downlink 15 norm',
                   'RS SNR/Antenna port - 1','RS SNR/Antenna port - 2',
                   'RSRP/Antenna port - 1','RSRP/Antenna port - 2',
                   'RS SNR/Antenna port - 1 full','RS SNR/Antenna port - 2 full',
                   'RSRP/Antenna port - 1 full','RSRP/Antenna port - 2 full',
                   'SE norm','SE']

    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data[column_list]

    # Remove outliers because of bandwidth normalization issues
    df['SE norm'][df['SE norm'] > 7.5] = np.nan

    # Bin RS-SNR to ]0,2],]2,4],...
    bins = np.arange(-10,50,2)
    ind = np.digitize(df['RS SNR/Antenna port - 1 full'].dropna(), bins,right=False)
    print(np.max(ind),np.min(ind))
    df['RS-SNR AP 1 round'] = df['RS SNR/Antenna port - 1 full']
    df['RS-SNR AP 1 round'].ix[df['RS SNR/Antenna port - 1 full'].dropna().index] = [bins[x]-2 for x in ind]

    plt.ion()

    rs_snr_groups = df[['Velocity','SE norm','RS-SNR AP 1 round','RS SNR/Antenna port - 1 full']].dropna().groupby('RS-SNR AP 1 round')  # sort=False
    # From http://stackoverflow.com/a/17302673
    for group_label,group_df in list(rs_snr_groups):
        print(group_label,group_df['SE norm'].median(),group_df['SE norm'].count(),
              group_df['Velocity'].median(),group_df['Velocity'].count())

    se_norm_per_rs_snr = rs_snr_groups['SE norm'].median().values
    velocity_per_rs_snr = rs_snr_groups['Velocity'].median().values
    rs_snr = rs_snr_groups.grouper.levels[0]

    print(se_norm_per_rs_snr)

    print('Spearman-r for SE norm versus Velocity (paired):')
    r = spearmanr(se_norm_per_rs_snr,velocity_per_rs_snr)
    n = len(se_norm_per_rs_snr)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    r = spearmanr(se_norm_per_rs_snr,rs_snr)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))

    plt.figure()
    ax_1 = plt.subplot2grid((2,1), (0,0))
    ax_1.plot(rs_snr,se_norm_per_rs_snr,c='m',lw=2.0,ls='-.')
    plt.grid(True)
    plt.xlabel('RS SNR AP 1 [dB]')
    plt.ylabel('Median of SE norm [bit/s/Hz]')
    ax_2 = ax_1.twinx()
    ax_2.bar(rs_snr,rs_snr_groups['SE norm'].count(),alpha=0.3)
    ax_1 = plt.subplot2grid((2,1), (1,0))
    dpl.plot_scatter(se_norm_per_rs_snr,velocity_per_rs_snr,
                     'SE norm','Velocity','bit/s/Hz','km/h',alpha=.8)
    # ax_2 = ax_1.twinx()
    # ax_2.bar(rs_snr,rs_snr_groups['Velocity'].count(),alpha=0.3)
    plt.figure()
    for group_label,group_df in list(rs_snr_groups):
        plt.boxplot(group_df['Velocity'].values,
                    positions=[group_df['SE norm'].median()],
                    widths=0.1)
    plt.xlim([0.2,0.65])
    plt.ylim([10,210])
    plt.grid()
    plt.xticks(np.arange(0,8,1))
    plt.xlabel('SE norm medians per RS-SNR AP 1')
    plt.ylabel('Velocity')
    plt.tight_layout()

    plt.figure()
    group_list = list(rs_snr_groups)
    for k in np.arange(0,5,1):
        for l in np.arange(0,4,1):
            # print(k*4+l,k,l)
            try:
                group_label,group_df = group_list[k*4+l]
                if len(group_df['SE norm']) < 10:
                    print('Not enough data')
                    continue
                plt.subplot2grid((5,4), (k,l))
                plt.plot(pacf(group_df['SE norm'].dropna(), nlags=20, method='ywunbiased', alpha=None),
                         lw=2.0,
                         label=group_label)
                plt.legend(loc=0,prop={'size':6})
                plt.ylim([-0.1,1.1])
                plt.xlim([0,20])
                plt.grid(True)
                plt.tight_layout()
            except (ValueError,IndexError):
                print('No valid data')

    plt.figure()
    # print(len(rs_snr_groups))
    group_list = list(rs_snr_groups)
    spearman_r_stat = [[],[],[],[]]
    for k in np.arange(0,5,1):
        for l in np.arange(0,4,1):
            try:
                group_label,group_df = group_list[k*4+l]
                if len(group_df['SE norm']) < 10:
                    print('Not enough data')
                    continue
                plt.subplot2grid((5,4), (k,l))
                plt.scatter(group_df['Velocity'],group_df['SE norm'],
                            s=10,
                            label=group_label)
                group_df['SE_norm'] = group_df['SE norm']
                model = glsar(formula='SE_norm ~ Velocity',data=group_df,missing='drop',rho=3)
                try:
                    fitted = model.iterative_fit(maxiter=7)
                    plt.plot(group_df['Velocity'],fitted.fittedvalues,c='r',lw=1.5)
                except (np.linalg.linalg.LinAlgError):
                    print('Not enough data for fit')
                plt.tick_params(axis='both', which='major', labelsize=7)
                plt.yticks([2,4,6,8])
                plt.xticks([50,150])
                plt.xlim([20,205])
                plt.ylim([0,8])
                plt.grid()
                plt.legend(loc=0,prop={'size':7})
                plt.tight_layout()
                r = spearmanr(group_df['Velocity'],group_df['SE norm'])
                ci_95 = ntp.spearmanr_ci_95(r[0],n)
                spearman_r_stat[0].append(r[0])
                spearman_r_stat[1].append(r[1])
                spearman_r_stat[2].append(np.absolute(ci_95[0]-r[0]))
                spearman_r_stat[3].append(np.absolute(ci_95[1]-r[0]))
                print(group_label,': ',r,' CI: ',ci_95)
            except (IndexError,TypeError,KeyError):
                print('No data')
    if args.print:
        plt.savefig(args.print,dpi=300,bbox_inches='tight')

    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    plt.bar(rs_snr[0:len(spearman_r_stat[0])],spearman_r_stat[0],
            yerr=[spearman_r_stat[2],spearman_r_stat[3]],
            color='m',alpha=.3)
    plt.xlim([-13,30])
    plt.grid()
    plt.xlabel('Group RS-SNR 1')
    plt.ylabel('Spearman R and 95% CI')
    plt.subplot2grid((2,1), (1,0))
    plt.bar(rs_snr[0:len(spearman_r_stat[1])],spearman_r_stat[1],
            color='Orange',alpha=.3)
    plt.grid()
    plt.ylim(0,0.2)
    plt.xlim([-13,30])
    plt.xlabel('Group RS-SNR 1')
    plt.xlabel('Test p-value')
    plt.tight_layout()
    input('press any key')

if __name__ == "__main__":
    args = setup_args()
    main(args)

