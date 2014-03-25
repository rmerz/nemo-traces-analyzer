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
    ntp.process_data(data,ntp.process_pdsch_rank)

    column_list = ['Velocity', 'Velocity full',
                   'valid_percentage',
                   'rank_1_per','rank_2_per',
                   'Requested rank - 1','Requested rank - 2']

    if args.select is None:
        df = tl.concat_pandas_data([df[column_list] for df in data ])
    else:
        df = data[column_list]

    rank_index = df[df['valid_percentage']>=99].index
    print(df['rank_1_per'][rank_index].describe())
    print(df['rank_2_per'][rank_index].describe())

    plt.ion()

    plt.subplot2grid((2,5), (0,0),rowspan=1,colspan=4)
    plt.title('Requested rank and effective rank percentages')
    req_rank = df[['Requested rank - 1','Requested rank - 2']].dropna()
    plt.plot(req_rank.index,req_rank.values[:,0],lw=0.2,color='LightBlue')
    plt.plot(req_rank.index,np.sum(req_rank.values[:],axis=1),c='Orange')
    plt.fill_between(req_rank.index,req_rank.values[:,0],0,color='LightBlue')
    plt.fill_between(req_rank.index,np.sum(req_rank.values[:],axis=1),req_rank.values[:,0],color='Orange')
    plt.ylim([0,102])
    plt.ylabel('Rank requested [%]')
    plt.subplot2grid((2,5), (0,4),rowspan=1)
    plt.boxplot((req_rank.values[:,0],req_rank.values[:,1]))
    plt.ylim([0,102])
    plt.grid(True)

    plt.subplot2grid((2,5), (1,0),rowspan=1,colspan=4)
    plt.plot(rank_index,df.rank_1_per[rank_index],lw=0.2,color='LightBlue')
    plt.plot(rank_index,df.rank_1_per[rank_index]+df.rank_2_per[rank_index],c='Orange')
    plt.fill_between(rank_index,
                     df.rank_1_per[rank_index].values,
                     0,
                     color='LightBlue')
    plt.fill_between(rank_index,
                     df.rank_1_per[rank_index].values+df.rank_2_per[rank_index].values,
                     df.rank_1_per[rank_index].values,
                     color='Orange')
    plt.ylim([0,102])
    plt.ylabel('Rank effective [%]')
    plt.subplot2grid((2,5), (1,4),rowspan=1)
    plt.boxplot((df.rank_1_per[rank_index],df.rank_2_per[rank_index]))
    plt.ylim([0,102])
    plt.grid(True)

    plt.tight_layout()
    if args.print:
        plt.savefig(args.print,dpi=300,bbox_inches='tight')

    input('press any key')

if __name__ == "__main__":
    args = setup_args()
    main(args)
