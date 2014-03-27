#!/usr/bin/env python

import sys, argparse

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from scipy.stats import probplot,spearmanr,kendalltau
import statsmodels.api as sm
from statsmodels.sandbox.tools.tools_pca import pcasvd
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
    parser.add_argument('--sample', type=int, help='Sample data-set to improve independence. Argument is percentage of data to keep.')
    parser.add_argument('--print', action='store_true', help='Print plots.')
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

    # Clean-up if need be
    df['SE norm'][df['SE norm']>8] = np.nan

    if args.sample is not None:
        plt.ion()
        df = ntp.sample_data(df,'SE norm',args.sample,choice=False,plot=True,pause=True)
        if args.print:
            plt.savefig('se_norm_acorr.png',dpi=300,bbox_inches='tight')

    print(df['Application throughput downlink'].describe())
    print(df['Application throughput downlink 10 norm'].describe())
    print(df['Application throughput downlink 15 norm'].describe())

    plt.ion()
    plt.figure()
    f_norm = lambda x: x/1e6
    x = np.arange(0,90,1)
    dpl.plot_ecdf_triplet(df['Application throughput downlink 10 norm'].dropna().apply(f_norm),
                          df['Application throughput downlink norm'].dropna().apply(f_norm),
                          df['Application throughput downlink 15 norm'].dropna().apply(f_norm),x,
                          'App. t. 10 MHz (PRB norm.)','App. t. All (PRB norm.)','App. t. 15 MHz (PRB norm.)','Mbit/s')

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
        plt.savefig('se_rs_snr_ap1_hist.png',dpi=300,bbox_inches='tight')

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
    if args.print:
        plt.savefig('se_rs_snr_ap2_hist.png',dpi=300,bbox_inches='tight')

    plt.figure()
    X = df[['Velocity','SE norm']].dropna()
    dpl.plot_scatter(X['Velocity'],
                     X['SE norm'],
                     'Velocity','Spectral efficiency',
                     'km/h','bit/s/Hz')
    if args.print:
        plt.savefig('se_velocity_scatter.png',dpi=300,bbox_inches='tight')

    plt.figure()
    dpl.plot_hist2d(X['Velocity'],
                     X['SE norm'],
                    'Velocity','Spectral efficiency',
                     'km/h','bit/s/Hz',bins=25)
    if args.print:
        plt.savefig('se_velocity_hist.png',dpi=300,bbox_inches='tight')

    plt.figure()
    X = df[['Velocity','RS SNR/Antenna port - 1']].dropna()
    dpl.plot_hist2d(X['Velocity'],
                    X['RS SNR/Antenna port - 1'],
                    'Velocity','RS-SNR AP 1',
                    'km/h','dB',bins=25)
    if args.print:
        plt.savefig('rs_snr_ap1_velocity_hist.png',dpi=300,bbox_inches='tight')

    plt.figure()
    X = df[['Velocity','RS SNR/Antenna port - 2']].dropna()
    dpl.plot_hist2d(X['Velocity'],
                    X['RS SNR/Antenna port - 2'],
                    'Velocity','RS-SNR AP 2',
                    'km/h','dB',bins=25)
    if args.print:
        plt.savefig('rs_snr_ap2_velocity_hist.png',dpi=300,bbox_inches='tight')

    # Correlation matrix: 
    # http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.corr.html
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    #
    # http://www.lisa.stat.vt.edu/sites/default/files/Non-Parametrics.pdf
    # On the p-value
    # http://www.graphpad.com/guides/prism/6/statistics/index.htm?stat_interpreting_results_correlati.htm
    # http://www.dummies.com/how-to/content/what-a-pvalue-tells-you-about-statistical-data.html
    # http://en.wikipedia.org/wiki/P-value
    df.rename(columns={'Application throughput downlink 10 norm':'App. t. 10',
                       'Application throughput downlink 15 norm':'App. t. 15',
                       'RS SNR/Antenna port - 1 full':'RS-SNR AP 1',
                       'RS SNR/Antenna port - 2 full':'RS-SNR AP 2',
                       'RSRP/Antenna port - 1 full':'RSRP AP 1',
                       'RSRP/Antenna port - 2 full':'RSRP AP 2'}, inplace=True)

    print(df[['SE norm',
              'Velocity',
              'RS-SNR AP 1','RS-SNR AP 2',
              'RSRP AP 1', 'RSRP AP 2']].dropna().corr(method='pearson'))
    print(df[['SE norm',
              'Velocity',
              'RS-SNR AP 1','RS-SNR AP 2',
              'RSRP AP 1', 'RSRP AP 2']].dropna().corr(method='spearman'))

    print('Spearman-r for SE norm (velocity, RS-SNR AP1/2, RSRP AP1/2):')
    X = df[['SE norm','Velocity']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['SE norm','RS-SNR AP 1']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['SE norm','RS-SNR AP 2']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['SE norm','RSRP AP 1']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['SE norm','RSRP AP 2']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))

    print('Kendall Tau for SE norm (velocity, RS-SNR AP1/2, RSRP AP1/2):')
    X = df[['SE norm','Velocity']].dropna().values
    print(kendalltau(X[:,0],X[:,1]))
    X = df[['SE norm','RS-SNR AP 1']].dropna().values
    print(kendalltau(X[:,0],X[:,1]))
    X = df[['SE norm','RS-SNR AP 2']].dropna().values
    print(kendalltau(X[:,0],X[:,1]))
    X = df[['SE norm','RSRP AP 1']].dropna().values
    print(kendalltau(X[:,0],X[:,1]))
    X = df[['SE norm','RSRP AP 2']].dropna().values
    print(kendalltau(X[:,0],X[:,1]))
    print('Spearman-r for velocity versus SE norm, RS-SNR (original), RSRP (original):')
    X = df[['Velocity','SE norm']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['Velocity','RS SNR/Antenna port - 1']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['Velocity','RS SNR/Antenna port - 2']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['Velocity','RSRP/Antenna port - 1']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))
    X = df[['Velocity','RSRP/Antenna port - 2']].dropna()
    n = len(X)
    r = spearmanr(X)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))

    # See http://www.r-bloggers.com/how-to-calculate-a-partial-correlation-coefficient-in-r-an-example-with-oxidizing-ammonia-to-make-nitric-a
    print('Partial correlation Velocity/SE controlling for RS-SNR AP 1 and RS-SNR AP 2:')
    Y0_Y1_X = df[['SE norm','Velocity','RS-SNR AP 1','RS-SNR AP 2']].dropna().values
    x = Y0_Y1_X[:,2:]
    y = Y0_Y1_X[:,0]
    resid_SE = sm.OLS(y,sm.add_constant(x)).fit().resid
    x = Y0_Y1_X[:,2:]
    y = Y0_Y1_X[:,1]
    resid_Velocity = sm.OLS(y,sm.add_constant(x)).fit().resid
    r = spearmanr(resid_SE,resid_Velocity)
    n = len(resid_SE)
    print(r)
    print('CI: ',ntp.spearmanr_ci_95(r[0],n))

    # Plot residuals
    plt.figure()
    plt.subplot2grid((2,1), (0,0))
    probplot(resid_SE, dist="norm", plot=plt)
    plt.grid()
    plt.tight_layout()
    plt.subplot2grid((2,1), (1,0))
    probplot(resid_Velocity, dist="norm", plot=plt)
    plt.grid()
    plt.tight_layout()

    # PCA
    X = df[['SE norm','Velocity','RS-SNR AP 1','RS-SNR AP 2']].dropna().values
    _,sdev,variance_proportion,variance_proportion_normed,cumulative_proportion,rotation,_ = ntp.pca_svd(X[:,1:],scale=True,print_debug=False)
    # _,factors,evals,_ = pcasvd(X[:,1:], keepdim=0, demean=True)
    print('PCA (scaled):')
    print(sdev)
    print('Variance proportion (and normed) and cumulative proportion:')
    print(variance_proportion,variance_proportion_normed,cumulative_proportion)
    print('Rotation:')
    print(rotation)

    print('PCA from sklearn (not scaled):')
    pca = PCA(n_components=None,whiten=False)  # Could be 2
    pca.fit(X[:,1:])
    print('Explained variance:')
    print(pca.explained_variance_ratio_)


    input('Press any key.')

if __name__ == "__main__":
    args = setup_args()
    main(args)
