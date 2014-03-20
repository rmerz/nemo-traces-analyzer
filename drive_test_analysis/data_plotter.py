"""A convenience wrapper around matplotlib combined with statsmodels and scipy.stats"""
import numpy as np
from statsmodels.distributions import ECDF
from scipy.stats import gaussian_kde

from matplotlib import pyplot as plt
import matplotlib.cm as cm

def plot_hist2d(data_0,data_1,label_0,label_1,
                 unit_0,unit_1,bins=50):
    # http://matplotlib.org/examples/color/colormaps_reference.html
    plt.hist2d(data_0.values,data_1.values,
               bins=bins,
               #cmap=cm.cool
               #cmap=cm.PuRd
               #cmap=cm.Purples
               cmap=cm.BuPu
               #cmap=cm.coolwarm
               #alpha=.3,
               )
    plt.grid(True)
    plt.colorbar()
    plt.xlabel(label_0+' [{}]'.format(unit_0))
    plt.ylabel(label_1+' [{}]'.format(unit_1))
    plt.tight_layout()

def plot_hist(data_0,label_0,
              unit_0,bins=50,
              alpha=.3):
    # http://matplotlib.org/examples/color/colormaps_reference.html
    plt.hist(data_0.values,
             bins=bins,
             color='Magenta',
             alpha=alpha,
             normed=True
         )
    plt.grid(True)
    plt.xlabel(label_0+' [{}]'.format(unit_0))
    plt.tight_layout()

def plot_scatter_pair(data_0a,data_1a,label_0a,label_1a,
                      data_0b,data_1b,label_0b,label_1b,
                      #unit_0,unit_1,
                      marker_size=20,alpha=.3):

    plt.scatter(data_0a,data_1a,
                s=marker_size,
                c='Orange',
                edgecolors='none',
                alpha=alpha,
                label=label_1a+' vs '+label_0a)

    plt.scatter(data_0b,data_1b,
                s=marker_size,
                c='m',
                edgecolors='none',
                alpha=alpha,
                label=label_1b+' vs '+label_0b)

    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=9)
    #plt.xlabel(label_0+' [{}]'.format(unit_0))
    #plt.ylabel(label_1+' [{}]'.format(unit_1))
    plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_scatter(data_0,data_1,label_0,label_1,
                 unit_0,unit_1,marker_size=20,alpha=.3):
    plt.scatter(data_0,data_1,
                s=marker_size,
                c='m',
                edgecolors='none',
                alpha=alpha,
                label=label_1+' vs '+label_0)

    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xlabel(label_0+' [{}]'.format(unit_0))
    plt.ylabel(label_1+' [{}]'.format(unit_1))
    plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_ts(data_0,label_0,unit,marker_size=10,ylim=None):
    plt.plot(data_0.index,data_0,
             marker='.',
             markersize=marker_size,
             c='m',
             label=label_0)

    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.ylabel(label_0+' [{}]'.format(unit))
    plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_ts_pair(data_0,data_1,label_0,label_1,unit,marker_size=10,ylim=None):
    plt.plot(data_0.index,data_0,
             marker='.',
             markersize=marker_size,
             c='m',
             label=label_0)

    plt.plot(data_1.index,data_1,
             marker='.',
             markersize=marker_size,
             c='Orange',
             label=label_1)

    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xlabel(label_0+'/'+label_1+' [{}]'.format(unit))
    plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_ecdf(data,x,label,unit):
    ecdf = ECDF(data.values)

    median = np.median(data.values)
    print(median)
    plt.plot(x,ecdf(x),
             lw=2.0,
             c='m',
             label=label+': median {:.1f} {}'.format(median,unit))
    plt.plot(x,0.5*np.ones(len(x)),
             lw=2.0,
             ls='--',
             c='b',
             alpha=.3)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xlabel(label+' [{}]'.format(unit))
    plt.ylabel('ECDF')
    plt.ylim([0,1.05])
    plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_ecdf_pair(data_0,
                   data_1,x,
                   label_0,label_1,
                   unit):
    ecdf = ECDF(data_0.values)
    median = np.median(data_0.values)
    print(median)
    plt.plot(x,ecdf(x),
             lw=2.0,
             c='m',
             label=label_0+': median {:.1f} {}'.format(median,unit))

    ecdf = ECDF(data_1.values)
    median = np.median(data_1.values)
    print(median)
    plt.plot(x,ecdf(x),
             lw=2.0,
             c='Orange',
             label=label_1+': median {:.1f} {}'.format(median,unit))

    plt.plot(x,0.5*np.ones(len(x)),
             lw=2.0,
             ls='--',
             c='b',
             alpha=.3)

    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xlabel(label_0+'/'+label_1+' [{}]'.format(unit))
    plt.ylabel('ECDF')
    plt.ylim([0,1.05])
    plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_ecdf_triplet(data_0,data_1,data_2,x,
                      label_0=None,label_1=None,label_2=None,
                      unit=None,plot_info=True):
    ecdf = ECDF(data_0.values)
    median = np.median(data_0.values)
    print(median)
    plt.plot(x,ecdf(x),
             lw=2.0,
             c='m',
             label=label_0+': median {:.1f} {}'.format(median,unit))

    ecdf = ECDF(data_1.values)
    median = np.median(data_1.values)
    print(median)
    plt.plot(x,ecdf(x),
             lw=2.0,
             c='Blue',
             label=label_1+': median {:.1f} {}'.format(median,unit))

    ecdf = ECDF(data_2.values)
    median = np.median(data_2.values)
    print(median)
    plt.plot(x,ecdf(x),
             lw=2.0,
             c='Orange',
             label=label_2+': median {:.1f} {}'.format(median,unit))

    plt.plot(x,0.5*np.ones(len(x)),
             lw=2.0,
             ls='--',
             c='b',
             alpha=.3)

    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=9)
    if plot_info:
        if label_0 is not None:
            plt.xlabel(label_0+'/'+label_1+'/'+label_2+' [{}]'.format(unit))
        else:
            plt.xlabel('[{}]'.format(unit))
        plt.ylabel('ECDF')
        plt.ylim([0,1.05])
        plt.legend(loc='upper left',prop={'size':10})
    plt.tight_layout()

def plot_density(data,x,label,unit):
    density = gaussian_kde(data)
    density.covariance_factor = lambda : .1
    density._compute_covariance()

    plt.plot(x,density(x),
             lw=2.0,
             c='m',
             label=label)
    plt.hist(data.values,bins=25,
             normed=1,
             color='b',
             alpha=.3)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xlabel(label+' [{}]'.format(unit))
    plt.ylabel('Density')
    plt.tight_layout()

def main():
    pass

if __name__ == "__main__":
    main()
