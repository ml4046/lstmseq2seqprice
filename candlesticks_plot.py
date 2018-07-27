"""
Plot functions for candlesticks
"""
import copy
import datetime
import itertools
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from matplotlib.dates import (
    DateFormatter, WeekdayLocator, DayLocator, MONDAY
)
from mpl_finance import candlestick_ohlc
import numpy as np
import pandas as pd
from collections import Counter

def get_cluster_ohlc(ohlc, date, window):
    i = ohlc.index.get_loc(date)
    return ohlc[i:i-window:-1][::-1]

def plot_candlesticks_cluster(ohlc, cluster, window_size, num_samples=5):
    """
    Plots multiple candlesticks of a given cluster
    Params:
        ohlc: df with date (as index), ohlc, cluster 
        num_samples: int number of samples to plot
    """
    if 'Cluster' not in ohlc.columns:
        raise AttributeError('clusters not found in ohlc')
    
    def get_cluster_ohlc(ohlc, date, window):
        i = ohlc.index.get_loc(date)
        return ohlc[i:i-window:-1][::-1]
    dates = ohlc.Cluster[ohlc.Cluster==cluster].index
    dates = np.random.choice(dates, num_samples)

    for date in dates:
        plot_candlesticks(get_cluster_ohlc(ohlc, date, window_size),figsize=(3,3))
        
def plot_candlesticks(data, figsize=(16,4)):
    """
    Plot a candlestick chart of the prices,
    appropriately formatted for dates
    """
    # Copy and reset the index of the dataframe
    # to only use a subset of the data for plotting
    df = copy.deepcopy(data)
    df.reset_index(inplace=True)
#     df['date_fmt'] = df.index
    df = df.rename(index=str,columns={'index':'Date'})
    df.Date = pd.to_datetime(df.Date)
    df['date_fmt'] = df['Date'].apply(
        lambda date: mdates.date2num(date.to_pydatetime())
    )
    
    # Set the axis formatting correctly for dates
    # with Mondays highlighted as a "major" tick
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter('%b %d %y')
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.2)
#     ax.xaxis.set_major_locator(mondays)
#     ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # Plot the candlestick OHLC chart using black for
    # up days and red for down days
    csticks = candlestick_ohlc(
        ax, df[
            ['date_fmt', 'Open', 'High', 'Low', 'Close']
        ].values, width=0.6, colorup= 'green', colordown='red')
    
    ax.xaxis_date()
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=45, horizontalalignment='right'
    )
    plt.show()
    
def plot_cluster_ordered_candles(data):
    """
    Plot a candlestick chart ordered by cluster membership
    with the dotted blue line representing each cluster
    boundary.
    """
    # Set the format for the axis to account for dates
    # correctly, particularly Monday as a major tick
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter("")
    fig, ax = plt.subplots(figsize=(16,4))
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # Sort the data by the cluster values and obtain
    # a separate DataFrame listing the index values at
    # which the cluster boundaries change
    df = copy.deepcopy(data)
    df.sort_values(by="Cluster", inplace=True)
    df.reset_index(inplace=True)
    df["clust_index"] = df.index
    df["clust_change"] = df["Cluster"].diff()
    change_indices = df[df["clust_change"] != 0]

    # Plot the OHLC chart with cluster-ordered "candles"
    csticks = candlestick_ohlc(
        ax, df[
            ["clust_index", 'Open', 'High', 'Low', 'Close']
        ].values, width=0.6,
        colorup='green', colordown='#ff0000'
    )

    # Add each of the cluster boundaries as a blue dotted line
    for row in change_indices.iterrows():
        plt.axvline(
            row[1]["clust_index"],
            linestyle="dashed", c="blue"
        )
    plt.xlim(0, len(df))
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=45, horizontalalignment='right'
    )
    plt.show()
    
def plot_3d_normalized_candles(data):
    """
    Plot a 3D scatterchart of the open-normalised bars
    highlighting the separate clusters by colour
    """
    labels = data.Cluster
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig, elev=21, azim=-136)
    ax.scatter(
        data["nHigh"], data["nLow"], data["nClose"],
        c=labels.astype(np.float)
    )
    ax.set_xlabel('High/Open')
    ax.set_ylabel('Low/Open')
    ax.set_zlabel('Close/Open')
    plt.show()
    
def plot_heatmap(cm, classes, cmap=plt.cm.Blues, show_prob=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Params:
        classes: list of unique classes
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Transition Matrix')
    plt.colorbar()
    classes = range(classes)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    if show_prob:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('t')
    plt.xlabel('t+n')
    plt.show()
    
def plot_signal_distribution(signal_distribution, cmap=plt.cm.Blues):
    """
    Plots heatmap for signal_distribution
    
    Params:
        classes: list of unique classes
        signal_distribution: pd DataFrame with distribution of signals in clusters
    """
    cm = signal_distribution.as_matrix()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Signal Distribution')
    #plt.colorbar()
    x_ticks = np.arange(len(signal_distribution.columns))
    y_ticks = np.arange(len(signal_distribution.index))
    plt.xticks(x_ticks, signal_distribution.columns)
    plt.yticks(y_ticks, signal_distribution.index)
    
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('trend/signal')
    plt.xticks(rotation=45)
    plt.xlabel('cluster')
    plt.show()