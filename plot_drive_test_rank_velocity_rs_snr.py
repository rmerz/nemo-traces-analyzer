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
    parser.add_argument('--print', nargs='+', help='Print figure to files.')
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
    ntp.process_data(data,ntp.process_velocity_round)
    ntp.process_data(data,ntp.process_pdsch_rank)
    ntp.process_data(data,ntp.process_lte_rs_snr)
    ntp.process_data(data,ntp.process_lte_rs_snr_full)
    ntp.process_data(data,ntp.process_lte_rs_snr_average_full)
    ntp.process_data(data,ntp.process_lte_rs_snr_average_full_round)

    column_list = ['Velocity',
                   'Velocity full','Velocity round',
                   'valid_percentage',
                   'rank_1_per','rank_2_per',
                   'Requested rank - 1','Requested rank - 2',
                   'RS SNR/Antenna port - 1 full','RS SNR/Antenna port - 2 full',
                   'RS SNR','RS SNR full','RS SNR full round']

    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data[column_list]

    rank_index = df['valid_percentage'].dropna().index.values
    print(df['rank_1_per'].iloc[rank_index].describe())
    print(df['rank_2_per'].iloc[rank_index].describe())

    plt.ion()
    # Velocity part
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    x = df['rank_1_per'].dropna()
    plt.scatter(df['Velocity full'].iloc[x.index],x,
                label='Rank 1')
    plt.xlim([30,201])
    plt.ylim([-1,101])
    plt.grid()
    plt.legend(loc='upper left')
    plt.ylabel('[%]')

    plt.subplot2grid((2,1), (1,0))
    x = df['rank_2_per'].dropna()
    plt.scatter(df['Velocity full'].iloc[x.index],x,
                color='m',label='Rank 2')
    plt.xlim([30,201])
    plt.ylim([-1,101])
    plt.grid()
    plt.legend(loc='upper left')
    plt.ylabel('[%]')
    plt.xlabel('Velocity [km/h]')
    plt.tight_layout()

    df_to_fit = df[['rank_1_per','rank_2_per','Velocity']].iloc[rank_index]
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    dpl.plot_hist2d(df_to_fit['Velocity'],
                    df_to_fit['rank_1_per'],
                    'Velocity','Rank 1',
                    'km/h','%',
                    bins=50)
    plt.subplot2grid((2,1), (1,0))
    dpl.plot_hist2d(df_to_fit['Velocity'],
                    df_to_fit['rank_2_per'],
                    'Velocity','Rank 2',
                    'km/h','%',
                    bins=50)
    plt.tight_layout()

    plt.figure()
    velocity_groups = df[['rank_1_per','rank_2_per','Velocity round']].iloc[rank_index].groupby('Velocity round')
    plt.subplot2grid((2,1), (0,0))
    for group_label,group_df in list(velocity_groups):
        print(group_label,group_df['rank_1_per'].median(),group_df['rank_1_per'].mean(),group_df['rank_1_per'].count())
        if group_df['rank_1_per'].count() > 20:
            plt.boxplot(group_df['rank_1_per'].values,
                        positions=[float(group_label)],
                        widths=5.0)
    plt.xlim([0,210])
    plt.xticks(np.arange(0,210,10))
    plt.ylim([0,101])
    plt.grid()
    plt.xlabel('Velocity (rounded)')
    plt.ylabel('Rank 1 percentage')
    plt.subplot2grid((2,1), (1,0))
    for group_label,group_df in list(velocity_groups):
        print(group_label,group_df['rank_2_per'].median(),group_df['rank_2_per'].mean(),group_df['rank_2_per'].count())
        if group_df['rank_2_per'].count() > 20:
            plt.boxplot(group_df['rank_2_per'].values,
                        positions=[float(group_label)],
                        widths=5.0)
    plt.xlim([0,210])
    plt.xticks(np.arange(0,210,10))
    plt.ylim([0,101])
    plt.grid()
    plt.xlabel('Velocity (rounded)')
    plt.ylabel('Rank 2 percentage')
    plt.tight_layout()
    if args.print:
        plt.savefig(args.print[0],dpi=300,bbox_inches='tight')

    # RS SNR part
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    x = df['rank_1_per'].dropna()
    plt.scatter(df['RS SNR full'].iloc[x.index],x,
                label='Rank 1')
    plt.xlim([-12,30])
    plt.ylim([-1,101])
    plt.grid()
    plt.legend(loc='upper left')
    plt.ylabel('[%]')
    plt.subplot2grid((2,1), (1,0))
    x = df['rank_2_per'].dropna()
    plt.scatter(df['RS SNR full'].iloc[x.index],x,
                color='m',label='Rank 2')
    plt.xlim([-12,30])
    plt.ylim([-1,101])
    plt.grid()
    plt.legend(loc='upper left')
    plt.ylabel('[%]')
    plt.xlabel('RS-SNR [dB]')
    plt.tight_layout()

    df_to_fit = df[['rank_1_per','rank_2_per','RS SNR full']].iloc[rank_index].dropna()
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    dpl.plot_hist2d(df_to_fit['RS SNR full'],
                    df_to_fit['rank_1_per'],
                    'RS SNR','Rank 1',
                    'dB','%',
                    bins=50)
    plt.subplot2grid((2,1), (1,0))
    dpl.plot_hist2d(df_to_fit['RS SNR full'],
                    df_to_fit['rank_2_per'],
                    'RS SNR','Rank 2',
                    'dB','%',
                    bins=50)
    plt.tight_layout()

    plt.figure()
    rs_snr_groups = df[['rank_1_per','rank_2_per','RS SNR full round']].iloc[rank_index].dropna().groupby('RS SNR full round')
    plt.subplot2grid((2,1), (0,0))
    for group_label,group_df in list(rs_snr_groups):
        print(group_label,group_df['rank_1_per'].median(),group_df['rank_1_per'].mean(),group_df['rank_1_per'].count())
        if group_df['rank_1_per'].count() > 20:
            plt.boxplot(group_df['rank_1_per'].values,
                        positions=[float(group_label)],
                        widths=0.9)
    plt.xlim([-12,30])
    plt.xticks(np.arange(-12,30,2))
    plt.ylim([0,101])
    plt.grid()
    plt.xlabel('RS SNR (rounded)')
    plt.ylabel('Rank 1 percentage')
    plt.subplot2grid((2,1), (1,0))
    for group_label,group_df in list(rs_snr_groups):
        print(group_label,group_df['rank_2_per'].median(),group_df['rank_2_per'].mean(),group_df['rank_2_per'].count())
        if group_df['rank_2_per'].count() > 20:
            plt.boxplot(group_df['rank_2_per'].values,
                        positions=[float(group_label)],
                        widths=0.9)
    plt.xlim([-12,30])
    plt.xticks(np.arange(-12,30,2))
    plt.ylim([0,101])
    plt.grid()
    plt.xlabel('RS SNR (rounded)')
    plt.ylabel('Rank 2 percentage')
    plt.tight_layout()
    if args.print:
        plt.savefig(args.print[1],dpi=300,bbox_inches='tight')

    input('press any key')

if __name__ == "__main__":
    args = setup_args()
    main(args)
