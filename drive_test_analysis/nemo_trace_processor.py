"""A series of functions to post-process trace files from NEMO"""
import logging
import numpy as np
import pandas as pd
# from pandas.tools.plotting import autocorrelation_plot
import matplotlib.mlab as mlab

from matplotlib import pyplot as plt

def sample_data(df,column_name,percentage,choice=True,plot=False,pause=False):
    valid_index = df[column_name].dropna().index.values
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html#numpy.random.choice
    if choice:
        sampled_index = np.random.choice(valid_index,size=len(valid_index)*(percentage/100),replace=False)
    else:
        sampled_index = [x for x in valid_index if np.random.rand() < percentage/100]
    # Validation part with the ACF plot
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        maxlags=100
        plt.acorr(df[column_name].ix[valid_index].dropna().values,
                  detrend=mlab.detrend_linear,
                  maxlags=maxlags,
                  lw=2.0,color='m')
        # plt.acorr(df[column_name].ix[valid_index].values,
        #           detrend=mlab.detrend_linear,
        #           maxlags=maxlags,
        #           lw=1.5,color='Orange',linestyle='--')
        # autocorrelation_plot(df[column_name].ix[valid_index])
        plt.ylim([-0.25,0.25])
        plt.xlim([-maxlags,maxlags])
        plt.grid(True)
        plt.xlabel('Full data')
        ax2 = fig.add_subplot(212)
        plt.acorr(df[column_name].ix[sampled_index].dropna().values,
                  detrend=mlab.detrend_linear,
                  maxlags=maxlags,
                  lw=2.0,color='m')  
        # plt.acorr(df[column_name].ix[sampled_index].values,
        #           detrend=mlab.detrend_linear,
        #           maxlags=maxlags,
        #           lw=1.5,color='Orange',linestyle='--')
        # autocorrelation_plot(df[column_name].ix[sampled_index])
        plt.ylim([-0.25,0.25])
        plt.xlim([-maxlags,maxlags])
        plt.grid(True)
        plt.xlabel('Sampled data: {} %'.format(percentage))
        plt.tight_layout()
    print('Number of samples before/after sampling: {}/{}'.format(len(valid_index),len(sampled_index)))
    if pause:
        input('Press any key to continue.')
    return df.ix[sampled_index]

def spearmanr_ci_95(r,n):
    """Calculate the 95% confidence interval for a spearman r coefficient

    See http://stats.stackexchange.com/a/18904
    """
    if n < 5:
        print('Cannot compute CI for Spearman R with less than five samples')
        return (np.nan,np.nan)
    delta = 1.96/np.sqrt(n-3)
    lower = np.tanh(np.arctanh(r)-delta)
    upper = np.tanh(np.arctanh(r)+delta)
    return (lower,upper)

def _non_zero_filter(df):
    if isinstance(df, pd.DataFrame):
        df['Velocity'] = df['Velocity'].astype(float)
        df['Velocity'] = df['Velocity'].interpolate(method='nearest')
        df = df[(np.isfinite(df['Velocity'])) & (df['Velocity']>0)]
    else:
        df = df.astype(float).interpolate(method='nearest')
        df = df[(np.isfinite(df)) & (df>0)]
    return df

def remove_non_positive_velocity_samples(data):
    if not isinstance(data,list):
        return _non_zero_filter(data)
    else:
        return [_non_zero_filter(df) for df in data]        

def process_data(data,func):
    pd.options.mode.chained_assignment = None  # default='warn'
    if not isinstance(data,list):
        func(data)
    else:
        [func(df) for df in data]
    pd.options.mode.chained_assignment = 'warn'  # default='warn'

def process_velocity(df):
    df['Velocity full'] = df['Velocity'].astype(float).interpolate(method='nearest')

def process_velocity_round(df):
    f_round_velocity = lambda x : np.ceil(x / 10)*10
    df['Velocity round'] = df['Velocity'].apply(f_round_velocity)

def process_lte_rename_mac_to_app(df):
    df.rename(columns={'MAC downlink throughput': 'Application throughput downlink'}, inplace=True)
    # print(df.columns)
    if 'RSRP/Antenna port - 1' not in list(df.columns):
        print('WARNING: no RSRP per antenna port.')
        df['RSRP/Antenna port - 1'] = df['RSRP (serving)']
        df['RSRP/Antenna port - 2'] = df['RSRP (serving)']

def process_lte_bw(df):
    df['DL bandwidth'].replace(to_replace=['20 MHz','15 MHz','10 MHz','n/a'],value= [20.0,15.0,10.0,np.nan],inplace=True)
    if len(df['DL bandwidth'].dropna()) == 1:
        # When we have only one sample because of a static lab test
        # and we never moved into another sector
        df['DL bandwidth full'] = df['DL bandwidth'].fillna(df['DL bandwidth'].dropna().astype(float).values)
    else:
        df['DL bandwidth full'] = df['DL bandwidth'].astype(float).interpolate(method='nearest')

def process_lte_prb_util(df):
    df['PRB utilization DL'].replace(to_replace=[' n/a','n/a'],value=np.nan,inplace=True)
    assert len(df['PRB utilization DL'].dropna()) > 0
    df['PRB utilization DL'] = df['PRB utilization DL'].astype(float)
    df['PRB utilization DL full'] = df['PRB utilization DL'].interpolate(method='nearest')

def process_lte_prb_util_interp(df):
    df['PRB utilization DL full nearest'] = df['PRB utilization DL'].interpolate(method='nearest')
    df['PRB utilization DL full linear'] = df['PRB utilization DL'].interpolate(method='linear')

def process_lte_pdcp_throughput(df):
    df['PDCP downlink throughput'].replace(to_replace=[' n/a','n/a'],value=np.nan,inplace=True)

def process_lte_app_throughput(df):
    df['Application throughput downlink'].replace(to_replace=[' n/a','n/a'],value=np.nan,inplace=True)

def process_lte_app_bw_prb_util(df):
    process_lte_bw(df)
    process_lte_prb_util(df)
    process_lte_app_throughput(df)

    # Normalize throughput with respect to PRB utilization
    f_norm = lambda x: x[0]/(x[1]/100)
    df['Application throughput downlink norm'] = df[['Application throughput downlink','PRB utilization DL full']].dropna().apply(f_norm,axis=1)

def _process_lte_prb_util_bw(df,bw):
    if np.any(df['DL bandwidth full']==bw):
        df['PRB utilization DL {}'.format(bw)] = df[df['DL bandwidth full']==bw]['PRB utilization DL']
    else:
        print('WARNING: no entry with {} MHz found.'.format(bw))
        df['PRB utilization DL {}'.format(bw)] = np.nan

def _process_lte_app_bw_prb_util_bw(df,bw):
    _process_lte_prb_util_bw(df,bw)
    df['Application throughput downlink {}'.format(bw)] = df[df['DL bandwidth full']==bw]['Application throughput downlink']
    # Upscale throughput with respect to PRB _utilization_ (this is
    # how much percentage of PRB we've been using): scale up to
    # hypothetical full bandwidth
    f_norm = lambda x: x[0]/(x[1]/100)

    temp_frame = df[['Application throughput downlink {}'.format(bw),'PRB utilization DL full']].dropna()
    if len(temp_frame)>0:
        df['Application throughput downlink {} norm'.format(bw)] = temp_frame.apply(f_norm,axis=1)
    else:
        df['Application throughput downlink {} norm'.format(bw)] = np.nan

def process_lte_app_bw_prb_util_bw10(df):
    _process_lte_app_bw_prb_util_bw(df,10)

def process_lte_app_bw_prb_util_bw15(df):
    _process_lte_app_bw_prb_util_bw(df,15)

def process_lte_app_bw_prb_util_bw20(df):
    _process_lte_app_bw_prb_util_bw(df,20)

def process_lte_prb_util_bw(df):
    _process_lte_prb_util_bw(df,10)
    _process_lte_prb_util_bw(df,15)
    _process_lte_prb_util_bw(df,20)

def process_lte_rs_snr(df):
    df['RS SNR/Antenna port - 1'] = df['RS SNR/Antenna port - 1'].replace(to_replace=[' n/a','n/a'],value=np.nan).astype(float)
    df['RS SNR/Antenna port - 2'] = df['RS SNR/Antenna port - 2'].replace(to_replace=[' n/a','n/a'],value=np.nan).astype(float)

def process_lte_rs_snr_full(df):
    df['RS SNR/Antenna port - 1 full'] = df['RS SNR/Antenna port - 1'].interpolate(method='nearest')
    df['RS SNR/Antenna port - 2 full'] = df['RS SNR/Antenna port - 2'].interpolate(method='nearest')

def process_lte_rs_snr_bw(df):
    df['RS SNR/Antenna port - 1 10'] = df[df['DL bandwidth full']==10]['RS SNR/Antenna port - 1']
    df['RS SNR/Antenna port - 1 15'] = df[df['DL bandwidth full']==15]['RS SNR/Antenna port - 1']

    df['RS SNR/Antenna port - 2 10'] = df[df['DL bandwidth full']==10]['RS SNR/Antenna port - 2']
    df['RS SNR/Antenna port - 2 15'] = df[df['DL bandwidth full']==15]['RS SNR/Antenna port - 2']

def process_lte_rsrp(df):
    df['RSRP (serving)'] = df['RSRP (serving)'].replace(to_replace=[' n/a','n/a'],value=np.nan).astype(float)
    if 'RSRP/Antenna port - 1' in list(df.columns):
        df['RSRP/Antenna port - 1'] = df['RSRP/Antenna port - 1'].replace(to_replace=[' n/a','n/a'],value=np.nan).astype(float)
        df['RSRP/Antenna port - 2'] = df['RSRP/Antenna port - 2'].replace(to_replace=[' n/a','n/a'],value=np.nan).astype(float)
    else:
        df['RSRP/Antenna port - 1'] = df['RSRP (serving)']
        df['RSRP/Antenna port - 2'] = df['RSRP (serving)']


def process_lte_rsrp_full(df):
    df['RSRP/Antenna port - 1 full'] = df['RSRP/Antenna port - 1'].interpolate()
    df['RSRP/Antenna port - 2 full'] = df['RSRP/Antenna port - 2'].interpolate()

def process_lte_rsrp_bw(df):
    df['RSRP (serving) 10'] = df[df['DL bandwidth full']==10]['RSRP (serving)']
    df['RSRP (serving) 15'] = df[df['DL bandwidth full']==15]['RSRP (serving)']

    df['RSRP/Antenna port - 1 10'] = df[df['DL bandwidth full']==10]['RSRP/Antenna port - 1']
    df['RSRP/Antenna port - 1 15'] = df[df['DL bandwidth full']==15]['RSRP/Antenna port - 1']

    df['RSRP/Antenna port - 2 10'] = df[df['DL bandwidth full']==10]['RSRP/Antenna port - 2']
    df['RSRP/Antenna port - 2 15'] = df[df['DL bandwidth full']==15]['RSRP/Antenna port - 2']

def process_lte_rsrp_rs_snr(df):
    process_lte_rsrp(df)
    process_lte_rs_snr(df)

def process_lte_rsrp_rs_snr_full(df):
    process_lte_rsrp_full(df)
    process_lte_rs_snr_full(df)

def process_lte_rs_snr_average_full(df):
    f_avg = lambda x: 10*np.log10((np.power(10,x[0]/10)+np.power(10,x[1]/10))/2)
    df['RS SNR'] = df[['RS SNR/Antenna port - 1','RS SNR/Antenna port - 2']].apply(f_avg,axis=1)
    df['RS SNR full'] = df[['RS SNR/Antenna port - 1 full','RS SNR/Antenna port - 2 full']].apply(f_avg,axis=1)

def process_lte_rs_snr_average_full_round(df):
    f_round = lambda x : np.round(x)
    df['RS SNR full round'] = df['RS SNR full'].apply(f_round)

def process_lte_rsrp_rs_snr_bw(df):
    process_lte_rsrp_bw(df)
    process_lte_rs_snr_bw(df)

def _get_pdsch_prb_max_index(df):
    return len(df.filter(regex='PDSCH PRB percentage').columns.tolist())

def _process_lte_prb_avg(df):
    pdsch_prb_max_index = _get_pdsch_prb_max_index(df)
    print(pdsch_prb_max_index)

    # Define columns of interest
    list_prb_percentage = list()
    list_prb_value = list()
    for index in range(1,pdsch_prb_max_index+1,1):
        list_prb_percentage.append('PDSCH PRB percentage - %d' % index)
        list_prb_value.append('PDSCH PRBs - %d' % index)

    prb_percentages = df[list_prb_percentage].fillna(0.0).values/100
    df['prb_percentages_sum'] = df[list_prb_percentage].fillna(0.0).sum(axis=1)
    prb_values = df[list_prb_value].fillna(0.0).values

    # print(df[df['prb_percentages_sum'] > 0]['prb_percentages_sum'])


    # This contains lots of zero values. Needs to be filtered out
    df['PRB Avg DL'] = np.sum(prb_percentages*prb_values,axis=1)
    df['PRB Avg DL full'] = df[df['prb_percentages_sum'] > 0]['PRB Avg DL'].astype(float)
    df['PRB Avg DL full'] = df['PRB Avg DL full'].interpolate(method='nearest')
    # print(df['PRB Avg DL full'].head(50))

def process_lte_se_rb(df):
    #print(df[['Application throughput downlink','PRB utilization DL full']].dropna().head(100))

    _process_lte_prb_avg(df)

    # Normalize throughput with respect to avg. number of PRB and bandwidth per PRB (180 kHz)
    f_norm_RB = lambda x: x[0]/x[1]
    f_norm_RB_Hz = lambda x: x/180e3

    df['SE RB'] = df[['Application throughput downlink','PRB Avg DL full']].dropna().apply(f_norm_RB,axis=1)
    df['SE RB'].replace(to_replace=[np.inf],value=np.nan,inplace=True)
    df['SE RB norm'] = df['SE RB'].dropna().apply(f_norm_RB_Hz)

    #print(df['SE RB'].dropna().describe())
    #print(df['SE RB norm'].dropna().describe())

def process_se_bw_norm(df):
    f_norm = lambda x : x/1e6/10
    se_10_norm = df['Application throughput downlink 10 norm'].apply(f_norm)
    df['SE 10 norm'] = se_10_norm
    df['SE 10 norm'].replace(to_replace=[np.inf],value=np.nan,inplace=True)

    f_norm = lambda x : x/1e6/15
    se_15_norm = df['Application throughput downlink 15 norm'].apply(f_norm)
    df['SE 15 norm'] = se_15_norm
    df['SE 15 norm'].replace(to_replace=[np.inf],value=np.nan,inplace=True)

    f_norm = lambda x : x/1e6/20
    se_20_norm = df['Application throughput downlink 20 norm'].apply(f_norm)
    df['SE 20 norm'] = se_20_norm
    df['SE 20 norm'].replace(to_replace=[np.inf],value=np.nan,inplace=True)

    df['SE norm'] = pd.concat([se_10_norm.dropna(), se_15_norm.dropna(), se_20_norm.dropna()]).reindex_like(df)
    # print(df['SE norm'].describe())

    se_10 = df['Application throughput downlink 10'].apply(f_norm)
    se_15 = df['Application throughput downlink 15'].apply(f_norm)
    se_20 = df['Application throughput downlink 15'].apply(f_norm)

    df['SE'] = pd.concat([se_10.dropna(), se_15.dropna()]).reindex_like(df)

def _get_pdsch_max_index(df):
    pdsch_max_index = len(df.filter(regex='PDSCH rank').columns.tolist())
    print(pdsch_max_index)
    return pdsch_max_index

def _get_pdsch_mcs_per(df,pdsch_max_index,rank_filter=None):
    list_pdsch_mcs_per = list()
    for index in range(1,pdsch_max_index+1,1):
        list_pdsch_mcs_per.append('PDSCH modulation percentage - %d' % index)

    # Get percentages for PDSCH MCS
    df[list_pdsch_mcs_per].replace(to_replace=[' n/a','n/a'],value=np.nan,inplace=True)

    if rank_filter is not None:
        df[list_pdsch_mcs_per] = df[list_pdsch_mcs_per].values*rank_filter

    df['valid_percentage'] = df[list_pdsch_mcs_per].sum(axis=1).dropna()
    pdsch_mcs_per = df[list_pdsch_mcs_per].fillna(0.0).values
    # This compact version removes all lines with non-valid percentage
    pdsch_mcs_per_compact = df[np.isfinite(df['valid_percentage'])][list_pdsch_mcs_per].fillna(0.0).values

    if rank_filter is not None:
        # pdsch_mcs_per = pdsch_mcs_per*rank_filter
        # Ugly but works
        df[list_pdsch_mcs_per] = df[list_pdsch_mcs_per].values*rank_filter
        pdsch_mcs_per_compact = df[np.isfinite(df['valid_percentage'])][list_pdsch_mcs_per].fillna(0.0).values
        # print(np.sum(pdsch_mcs_per_compact,axis=1))  # Debug
        # We want the column sum to be equal to 100: e.g. normalize
        l1_norm = np.sum(pdsch_mcs_per_compact,axis=1)
        pdsch_mcs_per_compact = 100*pdsch_mcs_per_compact/l1_norm.reshape(-1,1)
        # print(np.sum(pdsch_mcs_per_compact,axis=1))  # Debug

    return pdsch_mcs_per,pdsch_mcs_per_compact

def process_ber(df):
    df['MAC downlink BLER'] = df['MAC downlink BLER'].astype(float)
    df['MAC downlink BLER 1st'] = df['MAC downlink BLER 1st'].astype(float)
    df['MAC downlink BLER 2nd'] = df['MAC downlink BLER 2nd'].astype(float)
    df['MAC downlink BLER 3rd+'] = df['MAC downlink BLER 3rd+'].astype(float)
    df['MAC downlink residual BLER'] = df['MAC downlink residual BLER'].astype(float)

def _get_pdsch_rank_data(df,pdsch_max_index):
    list_pdsch_rank = list()
    for index in range(1,pdsch_max_index+1,1):
        list_pdsch_rank.append('PDSCH rank - %d' % index)

    # print(df['valid_percentage'].dropna())
    # print(pdsch_mcs_per.shape)

    # Extract rank information
    # To calculate rank 2
    data_rank = df[list_pdsch_rank]
    data_rank=data_rank.replace(to_replace=[0,1],value=0)
    data_rank=data_rank.replace(to_replace=[2],value=1)
    pdsch_rank_2 = data_rank.fillna(0.0).values

    #print(pdsch_rank_2[6662,:])
    #print(pdsch_mcs_per[6662,:])
    #print(df['rank_2_per'].iloc[6662])

    # To calculate rank 1
    data_rank = df[list_pdsch_rank]
    data_rank=data_rank.replace(to_replace=[0],value=1)
    data_rank=data_rank.replace(to_replace=[2],value=0)
    pdsch_rank_1 = data_rank.fillna(0.0).values

    return pdsch_rank_1,pdsch_rank_2

def process_pdsch_rank(df):
    pdsch_max_index = _get_pdsch_max_index(df)

    pdsch_rank_1,pdsch_rank_2 = _get_pdsch_rank_data(df,pdsch_max_index)

    pdsch_mcs_per,_ = _get_pdsch_mcs_per(df,pdsch_max_index)  # df['valid_percentage'] is now available

    df['rank_2_per'] = np.sum(pdsch_mcs_per*pdsch_rank_2,axis=1)
    df['rank_1_per'] = np.sum(pdsch_mcs_per*pdsch_rank_1,axis=1)

def _pre_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per):
    list_mcs_0 = [] # Codeword 0
    for index in range(1,pdsch_max_index+1,1):
        list_mcs_0.append('PDSCH MCS index for codeword 0 - %d' % index)

    # Replace with -1 for MCS that we don't know about
    df[list_mcs_0] = df[list_mcs_0].replace(to_replace=[' n/a','n/a'],value=-1)

    # Note: MCS index goes from 0 to 28 with 29 to 31 being reserved
    data_mcs_0 = df[np.isfinite(df['valid_percentage'])][list_mcs_0].astype(float)  # astype(float) is very important

    # Big fat matrix
    mcs_per_0 = np.zeros([33,len(df['valid_percentage'].dropna())])

    assert len(data_mcs_0) == mcs_per_0.shape[1]
    assert len(data_mcs_0) == len(pdsch_mcs_per)
    # print('Check')
    # print(len(data_mcs_0))
    # print(mcs_per_0.shape[1])
    # print(mcs_per_0.shape)
    # print(len(df['valid_percentage'].dropna()),len(df['valid_percentage']))
    # print(df['valid_percentage'].index.values[-10:])
    # print(df[list_mcs_0].index.values[-10:])

    for mcs_val in range(0,32,1):
        # print('MCS %d' % mcs_val)
        mcs_per_0[mcs_val,:] = np.sum(pdsch_mcs_per*(data_mcs_0[data_mcs_0 == mcs_val].replace(to_replace=[mcs_val],value=1).fillna(0.0).values),axis=1)
    mcs_per_0[32,:] = np.sum(pdsch_mcs_per*(data_mcs_0[data_mcs_0 == -1].replace(to_replace=[-1],value=1).fillna(0.0).values),axis=1)
    # print(np.sum(mcs_per_0,axis=0))  # Debug
    return mcs_per_0


def _get_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per):
    mcs_per_0 = _pre_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

    # See table 7.1.7.1-1 in 3GPP document
    df['mcs_q_2'] = np.nan
    df['mcs_q_2'].iloc[df['valid_percentage'].dropna().index] = np.sum(mcs_per_0[0:10,:],axis=0)  # Q_m = 2
    df['mcs_q_4'] = np.nan
    df['mcs_q_4'].iloc[df['valid_percentage'].dropna().index] = np.sum(mcs_per_0[10:17,:],axis=0)  # Q_m = 4
    df['mcs_q_6'] = np.nan
    df['mcs_q_6'].iloc[df['valid_percentage'].dropna().index] = np.sum(mcs_per_0[17:29,:],axis=0)  # Q_m = 6
    df['mcs_reserved'] = np.nan
    df['mcs_reserved'].iloc[df['valid_percentage'].dropna().index] = np.sum(mcs_per_0[29:32,:],axis=0)  # Reserved
    df['mcs_na'] = np.nan
    df['mcs_na'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[32,:]  # MCS n/a

def _get_pdsch_mcs_qpsk(df,pdsch_max_index,pdsch_mcs_per):
    mcs_per_0 = _pre_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

    # Need to normalize
    normalization_factor = np.sum(mcs_per_0[0:11,:],axis=0)/100

    df['mcs_0'] = np.nan
    df['mcs_0'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[0,:]/normalization_factor
    df['mcs_1'] = np.nan
    df['mcs_1'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[1,:]/normalization_factor
    df['mcs_2'] = np.nan
    df['mcs_2'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[2,:]/normalization_factor
    df['mcs_3'] = np.nan
    df['mcs_3'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[3,:]/normalization_factor
    df['mcs_4'] = np.nan
    df['mcs_4'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[4,:]/normalization_factor
    df['mcs_5'] = np.nan
    df['mcs_5'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[5,:]/normalization_factor
    df['mcs_6'] = np.nan
    df['mcs_6'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[6,:]/normalization_factor
    df['mcs_7'] = np.nan
    df['mcs_7'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[7,:]/normalization_factor
    df['mcs_8'] = np.nan
    df['mcs_8'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[8,:]/normalization_factor
    df['mcs_9'] = np.nan
    df['mcs_9'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[9,:]/normalization_factor

def _get_pdsch_mcs_16qam(df,pdsch_max_index,pdsch_mcs_per):
    mcs_per_0 = _pre_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

    # Need to normalize
    normalization_factor = np.sum(mcs_per_0[10:17,:],axis=0)/100

    df['mcs_10'] = np.nan
    df['mcs_10'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[10,:]/normalization_factor
    df['mcs_11'] = np.nan
    df['mcs_11'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[11,:]/normalization_factor
    df['mcs_12'] = np.nan
    df['mcs_12'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[12,:]/normalization_factor
    df['mcs_13'] = np.nan
    df['mcs_13'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[13,:]/normalization_factor
    df['mcs_14'] = np.nan
    df['mcs_14'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[14,:]/normalization_factor
    df['mcs_15'] = np.nan
    df['mcs_15'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[15,:]/normalization_factor
    df['mcs_16'] = np.nan
    df['mcs_16'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[16,:]/normalization_factor

def _get_pdsch_mcs_64qam(df,pdsch_max_index,pdsch_mcs_per):
    mcs_per_0 = _pre_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

    # Need to normalize
    normalization_factor = np.sum(mcs_per_0[17:29,:],axis=0)/100

    df['mcs_17'] = np.nan
    df['mcs_17'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[17,:]/normalization_factor
    df['mcs_18'] = np.nan
    df['mcs_18'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[18,:]/normalization_factor
    df['mcs_19'] = np.nan
    df['mcs_19'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[19,:]/normalization_factor
    df['mcs_20'] = np.nan
    df['mcs_20'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[20,:]/normalization_factor
    df['mcs_21'] = np.nan
    df['mcs_21'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[21,:]/normalization_factor
    df['mcs_22'] = np.nan
    df['mcs_22'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[22,:]/normalization_factor
    df['mcs_23'] = np.nan
    df['mcs_23'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[23,:]/normalization_factor
    df['mcs_24'] = np.nan
    df['mcs_24'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[24,:]/normalization_factor
    df['mcs_25'] = np.nan
    df['mcs_25'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[25,:]/normalization_factor
    df['mcs_26'] = np.nan
    df['mcs_26'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[26,:]/normalization_factor
    df['mcs_27'] = np.nan
    df['mcs_27'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[27,:]/normalization_factor
    df['mcs_28'] = np.nan
    df['mcs_28'].iloc[df['valid_percentage'].dropna().index] = mcs_per_0[28,:]/normalization_factor

def process_pdsch_mcs_rank_1(df):
    pdsch_max_index = _get_pdsch_max_index(df)
    pdsch_rank_1,_ = _get_pdsch_rank_data(df,pdsch_max_index)

    _,pdsch_mcs_per = _get_pdsch_mcs_per(df,pdsch_max_index,pdsch_rank_1)

    _get_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

def process_pdsch_mcs_rank_2(df):
    pdsch_max_index = _get_pdsch_max_index(df)
    _,pdsch_rank_2 = _get_pdsch_rank_data(df,pdsch_max_index)

    _,pdsch_mcs_per = _get_pdsch_mcs_per(df,pdsch_max_index,pdsch_rank_2)

    _get_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

def process_pdsch_mcs(df):
    pdsch_max_index = _get_pdsch_max_index(df)
    _,pdsch_mcs_per = _get_pdsch_mcs_per(df,pdsch_max_index)

    _get_pdsch_mcs(df,pdsch_max_index,pdsch_mcs_per)

def process_pdsch_mcs_qpsk(df):
    pdsch_max_index = _get_pdsch_max_index(df)
    _,pdsch_mcs_per = _get_pdsch_mcs_per(df,pdsch_max_index)

    _get_pdsch_mcs_qpsk(df,pdsch_max_index,pdsch_mcs_per)

def process_pdsch_mcs_16qam(df):
    pdsch_max_index = _get_pdsch_max_index(df)
    _,pdsch_mcs_per = _get_pdsch_mcs_per(df,pdsch_max_index)

    _get_pdsch_mcs_16qam(df,pdsch_max_index,pdsch_mcs_per)

def process_pdsch_mcs_64qam(df):
    pdsch_max_index = _get_pdsch_max_index(df)
    _,pdsch_mcs_per = _get_pdsch_mcs_per(df,pdsch_max_index)

    _get_pdsch_mcs_64qam(df,pdsch_max_index,pdsch_mcs_per)

def _extract_info(w_sorted,v_sorted,
                  print_debug=False):
    # sdev: see http://stat.ethz.ch/R-manual/R-patched/library/stats/html/prcomp.html
    sdev = np.sqrt(w_sorted)
    if print_debug:
        sys.stdout.write('sdev: {}\n'.format(sdev))
    # see https://stat.ethz.ch/pipermail/r-help/2005-July/075040.html
    variance_proportion = np.power(sdev,2.0)
    variance_proportion_normed = variance_proportion/np.sum(variance_proportion)
    cumulative_proportion = np.cumsum(variance_proportion/np.sum(variance_proportion))
    # Rotation/Loading: see http://stat.ethz.ch/R-manual/R-patched/library/stats/html/prcomp.html
    rotation = v_sorted
    if print_debug:
        sys.stdout.write('rotation:\n{}\n'.format(rotation))
    # Unrotated component matrix according to http://stats.stackexchange.com/a/17102
    component_matrix = np.sqrt(w_sorted)*v_sorted
    if print_debug:
        sys.stdout.write('Unrotated component matrix\n{}\n'.format(component_matrix))
    communalities = np.sum(np.power(component_matrix,2.0),axis=1)
    if print_debug:
        sys.stdout.write('Communalities\n{}\n'.format(communalities))
    return (sdev,variance_proportion,variance_proportion_normed,cumulative_proportion,rotation,component_matrix,communalities)

def _apply_transformation(data,w_sorted,v_sorted,
                         print_debug=False):
    pca_output = np.dot(v_sorted.T,data.T).T
    if print_debug:
        sys.stdout.write('PCA output:\n{}\n'.format(pca_output))
    return pca_output

def _sort_and_reduce(w,v,k=None,print_debug=False):
    # Sort by eigenvalues (see http://stats.stackexchange.com/questions/17090/pca-and-fa-example-calculation-of-communalities)
    index_order = np.argsort(-w)
    w_sorted = w[index_order]
    v_sorted = v[:, index_order]
    if print_debug:
        sys.stdout.write('Sorted eigenvalues: {}\nSorted eigenvectors\n{}\n'.format(w_sorted,v_sorted))
    if k is not None:
        assert k <= len(w)
        w_sorted = w_sorted[:k]
        v_sorted = v_sorted[:,:k]
        if print_debug:
            sys.stdout.write('Sorted selected eigenvalues: {}\nSorted selected eigenvectors\n{}\n'.format(w_sorted,v_sorted))
    return w_sorted,v_sorted

def pca_svd(data,k_comp=None,scale=False,print_debug=False):
    """Implements PCA with SVD"""
    if print_debug:
        sys.stdout.write('{}\n'.format(data))
    data_centered = data - np.mean(data,axis=0)
    if scale:
        m = np.corrcoef(data_centered.T)
    else:
        m = np.cov(data_centered.T)
    # Singular value decomposition
    u,s,v = np.linalg.svd(m,full_matrices=False)

    s,u = _sort_and_reduce(s,u,k=k_comp,
                           print_debug=print_debug)
    sdev,variance_proportion,variance_proportion_normed,cumulative_proportion,rotation,component_matrix,communalities = _extract_info(s,u,print_debug=print_debug)
    pca_output = _apply_transformation(data_centered,s,u,
                                       print_debug=print_debug)
    return (pca_output,sdev,variance_proportion,variance_proportion_normed,cumulative_proportion,rotation,communalities)


def main():
    pass

if __name__ == "__main__":
    main()
