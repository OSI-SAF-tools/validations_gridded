import os
from glob import glob
from os.path import join, isdir
import platform

import matplotlib.pylab as plt
import matplotlib.dates as mdates

import numpy as np
import xarray as xr
import yaml


years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

def get_config():
    with open("config.yml", 'r') as stream:
        base_url = yaml.load(stream)
    return base_url


def plot_bar(da20pct, da10pct, fig, ax):
    attrs = da10pct.attrs
    plt.bar(da20pct['time'].to_series(), da20pct.values, width=6, color='tab:red')
    plt.bar(da10pct['time'].to_series(), da10pct.values, width=6, color='tab:blue')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.legend(['Within 20%', 'Within 10%'], loc=4)
    ax.set_xlabel('{0}'.format(attrs['x_label']))
    ax.set_ylabel('{0} ({1})'.format(attrs['y_label'], attrs['unit']))
    ax.grid()
    # # fig.autofmt_xdate()()()
    fig.tight_layout()
    return fig, ax


def plot_line(da, fig, ax, color=None):
    plt.plot(da['time'].to_series(), da, '-bo', ms=2, lw=0.8, c=color)
    return fig, ax


def plots_conc(ds, save_dir):
    for v in ['bias', 'stddev']:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6 / 1.618))
        fig, ax = plot_line(ds['ice_' + v], fig, ax)
        fig, ax = plot_line(ds['water_' + v], fig, ax, color='tab:green')
        fig, ax = plot_line(ds['intermediate_' + v], fig, ax, color='grey')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        attrs = ds['ice_' + v].attrs
        ax.set_xlabel('{0}'.format(attrs['x_label']))
        ax.set_ylabel('{0} ({1})'.format(attrs['y_label'], attrs['unit']))
        ax.legend(['Ice', 'Water', 'Intermediate'])
        ax.grid()
        ax.axhline(color='k')
        # # fig.autofmt_xdate()()()
        fig.tight_layout()
        plt.savefig(join(save_dir, v + '.png'), dpi=180)
        fig.clf()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 / 1.618))
    plot_bar(ds['within_20pct'], ds['within_10pct'], fig, ax)
    fig.tight_layout()
    plt.savefig(join(save_dir, 'within_pct' + '.png'), dpi=180)
    fig.clf()

    vs = [v for v in ds.data_vars if '6monthly' in v]
    if vs:
        ds[vs].to_dataframe().to_csv(join(save_dir, '6monthly_ave.csv'))


def plot_bar_edge(true_ice, false_water, false_ice, fig, ax):
    total = (true_ice + false_water + false_ice) / 100
    true_ice /= total
    false_water /= total
    false_ice /= total
    width = 8
    plt.bar(true_ice['time'].to_series(), true_ice, width=width, color='gray')
    plt.bar(false_water['time'].to_series(), false_water,
            bottom=true_ice, width=width, color='tab:blue')
    plt.bar(false_ice['time'].to_series(), false_ice,
            bottom=(true_ice + false_water), width=width, color='tab:red')
    attrs = ds.true_ice.attrs
    ax.set_ylim([0, 100])
    ax.set_xlabel('{0}'.format(attrs['x_label']))
    ax.set_ylabel('{0}'.format(attrs['y_label']))
    ax.legend(['True Ice', 'False Water', 'False Ice'], loc=4)

    ax.grid()
    # # fig.autofmt_xdate()()()
    fig.tight_layout()
    return fig, ax


def plots_edge(ds, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 / 1.618))
    plot_bar_edge(ds['true_ice'].astype(np.float),
                  ds['false_water'].astype(np.float),
                  ds['false_ice'].astype(np.float), fig, ax)
    plt.savefig(join(save_dir, 'pixel_true_ice' + '.png'), dpi=180)
    fig.clf()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 / 1.618))
    plot_line(ds['distance_between_edges_mean'], fig, ax)
    ax.axhline(color='k')
    ax.grid()
    attrs = ds['distance_between_edges_mean'].attrs
    ax.set_xlabel('{0}'.format(attrs['x_label']))
    ax.set_ylabel('{0} ({1})'.format(attrs['y_label'], attrs['unit']))
    plt.savefig(join(save_dir, 'distance_between_edges' + '.png'), dpi=180)
    fig.clf()
    vs = [v for v in ds.data_vars if '6monthly' in v]
    if vs:
        ds[vs].to_dataframe().to_csv(join(save_dir, '6monthly_ave.csv'))
    return


if __name__ == '__main__':
    config = get_config()
    direc = join(config['MachineConfigs'][platform.node()]['results'])
    for fname in glob(direc + '*.nc'):
        print(fname)
        save_dir = fname.replace('.nc', '')
        # if not isdir(save_dir):
        if True:
            try:
                os.mkdir(save_dir)
            except OSError:
                pass
            ds = xr.open_dataset(fname)
            try:
                if 'conc' in fname:
                    plots_conc(ds, save_dir)
                elif 'edge' in fname:
                    plots_edge(ds, save_dir)
            except KeyError:
                pass
