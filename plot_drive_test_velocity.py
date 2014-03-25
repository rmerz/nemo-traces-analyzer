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
        logging.debug('Remove non-positive velocity samples')
        data = ntp.remove_non_positive_velocity_samples(data)

    if args.select is None:
        df = tl.concat_pandas_data([df['Velocity'] for df in data ])
    else:
        df = data['Velocity']

    print(df.describe())

    plt.ion()
    plt.figure()
    dpl.plot_ts(df,'Velocity','km/h',marker_size=2,ylim=None)
    if args.print:
        plt.savefig(args.print[0],dpi=300,bbox_inches='tight')

    plt.figure()
    x = np.arange(0,210,1)
    plt.subplot2grid((2,1),(0,0))
    dpl.plot_hist(df.dropna().astype(float),'Velocity','km/h',bins=20,normed=False)
    plt.xlim([np.min(x),np.max(x)])

    plt.subplot2grid((2,1),(1,0))
    dpl.plot_ecdf(df.dropna().astype(float),x,'Velocity','km/h')
    plt.xlim([np.min(x),np.max(x)])
    if args.print:
        plt.savefig(args.print[1],dpi=300,bbox_inches='tight')

    plt.figure()
    x = np.arange(-5,220,10)
    dpl.plot_hist(df.dropna().astype(float),'Velocity','km/h',bins=x,normed=False)
    plt.xlim([np.min(x),np.max(x)])
    if args.print:
        plt.savefig(args.print[2],dpi=600,bbox_inches='tight')

    input('Press any key')

if __name__ == "__main__":
    args = setup_args()
    main(args)

