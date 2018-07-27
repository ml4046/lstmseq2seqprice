"""
Utils for data retriving, processing etc.
"""
import numpy as np
import pandas as pd
import ccxt
import itertools
import ta_utils as ta
import matplotlib.ticker as ticker
import datetime as datetime
from collections import Counter

def get_ohlc(symbol, since=None, timeframe='1d', exchange=None):
    """
    Returns ohlc data for symbol from bitmex via. ccxt
    Params:
        symbol: str crypto pair
        since: str datetime format '2017-01-01'
        timeframe: str ohlc time window to be given in 
    Returns:
        ohlc: pd DataFrame ohlc data of symbol
    """
    ohlc = []
    exchange = ccxt.bitmex() if exchange is None else exchange
    if since != None:
        since = to_timestamp(pd.to_datetime(since))
        curr_ohlc = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=450, since=since)
    else:
        curr_ohlc = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=450)
        
    count = 1
    while (len(curr_ohlc) == 449) or (len(curr_ohlc)==450):
        print 'retrieving iteration %d' % count
        print len(curr_ohlc)
        ohlc.append(pd.DataFrame(curr_ohlc, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']).iloc[:-1])
        curr_ohlc = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=450, since=curr_ohlc[-1][0])
        count += 1
    ohlc.append(pd.DataFrame(curr_ohlc, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']))  
    ohlc = pd.concat(ohlc, ignore_index=True)
    ohlc.Date = pd.to_datetime(ohlc.Date, unit='ms')
    return ohlc

def generate_features(ohlc, with_prices=False):
    """
    Generates the following features for given OHLC(V)
    
    General: Daily OHLCV
    Volatitilty: perBollinger20, ATR14
    Trend: MA5, MA15 , EMA20, MACD12-26-9, CCI20, ROC
    Volume: ADI, OBV, CMF20
    Momentum: RSI14, SMI14, WVAD
    
    Param:
        Open: np array of Open (vice versa for others)
    Return:
        X: pd DataFrame of features
    """
    Open, high, low, close, volume = ohlc.Open.as_matrix(), ohlc.High.as_matrix(), \
        ohlc.Low.as_matrix(), ohlc.Close.as_matrix(), ohlc.Volume.as_matrix()
    #generating features
    perBoll20 = pd.Series(ta.perBollinger(close), name='perBoll20')
    ATR = pd.Series(ta.ATR(high, low, close), name='ATR14')
    
    MA5 = pd.Series(ta.ma(close, 5),name='MA5')
    MA15 = pd.Series(ta.ma(close, 15), name='MA15')
    EMA20 = pd.Series(ta.ema(close, 20), name='EMA20')
    MACD = pd.Series(ta.MACD(close), name='MACD12-26-9')
    CCI20 = pd.Series(ta.CCI(high, low, close), name='CCI20')
    ROC = pd.Series(ta.rate_of_change(close), name='ROC')
    
    ADI = pd.Series(ta.ADI(high, low, close, volume), name='ADI')
    OBV = pd.Series(ta.OBV(close, volume), name='OBV')
    CMF20 = pd.Series(ta.CMF(high, low, close, volume), name='CMF20')
    
    RSI14 = pd.Series(ta.RSI(close), name='RSI14')
    SMI14 = pd.Series(ta.SMI(high, low, close), name='SMI14')
    WVAD = pd.Series(ta.WVAD(Open, high, low, close, volume), name='WVAD')
    
    if with_prices:
        Open = pd.Series(Open, name='Open')
        high = pd.Series(high, name='high')
        low = pd.Series(low, name='low')
        close = pd.Series(close, name='close')
        
        features = pd.concat([perBoll20, ATR, MA5, MA15, EMA20, MACD, CCI20, ROC,
                          ADI, OBV, CMF20, RSI14, SMI14, WVAD,
                          Open, high, low, close], axis=1)
        for col in features.columns:
            features[col] = features[col].shift(sum(features[col].isna()))
        return features        

    features = pd.concat([perBoll20, ATR, MA5, MA15, EMA20, MACD, CCI20, ROC,
                      ADI, OBV, CMF20, RSI14, SMI14, WVAD], axis=1)
    features.index = ohlc.index
    for col in features.columns:
        features[col] = features[col].shift(sum(features[col].isna()))
    return features

def transition_matrix(clusters, K, future_step=5):
    """
    Calculates transition prof i -> j for rolling window given future_step lookahead
    Params:
        clusters: np array
        future_step: int number of steps ahead to observe (still uses rolling window)
    Return:
        dist_mat: np ndarray
    """
    today = pd.Series(clusters,name='today')
    lag = pd.Series(clusters,name='lag').shift(future_step)
    t = pd.concat([lag,today], axis=1).dropna().astype(int)
    counts = Counter(list(zip(t.lag.as_matrix(),t.today.as_matrix()))) #creates count for tuple (i,j)
    dist_mat = np.zeros((K,K))
    for i,j in counts.keys():
        dist_mat[i][j] = counts[(i,j)]
    return dist_mat/dist_mat.sum(axis=0)[:,None]

def transition_matrix_split_window(clusters, window):
    """
    Calculates transition prob. from i to j for split window (not rolling window)
    """
    today = pd.Series(clusters,name='today')
    K = len(pd.unique(today))
    lag = pd.Series(clusters,name='lag').shift(window)
    t = pd.concat([lag,today], axis=1).dropna().astype(int)
    counts = Counter(list(zip(t.lag.as_matrix()[::window],t.today.as_matrix()[::window]))) #creates count for tuple (i,j)
    dist_mat = np.zeros((K,K))
    for i,j in counts.keys():
        dist_mat[i][j] = counts[(i,j)]
    return dist_mat/dist_mat.sum(axis=0)[:,None]
    
def to_timestamp(datetime):
    unix_epoch = np.datetime64(0, 'ms')
    one_msecond = np.timedelta64(1, 'ms')
    return int((datetime - unix_epoch) / one_msecond)

def n_day_lag(ohlc, n=6):
    """
    O_t / C_{t-1}
    """
    n -= 1
    ohlc_rlag = ohlc.div(ohlc.Close.shift(1),axis=0)
    col_names = [('Open_lag'+str(i),'High_lag'+str(i),'Low_lag'+str(i),'Close_lag'+str(i)) for i in range(n)]
    col_names = list(itertools.chain(*col_names))
    lags = pd.concat([ohlc_rlag.shift(i) for i in range(n)], axis=1)
    lags.columns = col_names
    return lags

def n_day_ohlc_lag(ohlc, n=6):
    """
    Returns lag for unadjusted prices
    """
    ohlc_rlag = ohlc
    col_names = [('Open_lag'+str(i),'High_lag'+str(i),'Low_lag'+str(i),'Close_lag'+str(i)) for i in range(n)]
    col_names = list(itertools.chain(*col_names))
    lags = pd.concat([ohlc_rlag.shift(i) for i in range(n)], axis=1)
    lags.columns = col_names
    return lags

def get_next_n_day_close(ohlc, n=3):
    col_names = ['Close_n+'+str(i+1) for i in range(n)]
    lags = pd.concat([ohlc.Close.shift(-(i+1)) for i in range(n)], axis=1)
    lags.columns = col_names
    return lags

def get_candlestick_signals(ohlc, window_size):
    all_signals = {}
    for i in range(len(ohlc)-window_size+1):
        curr = ohlc.iloc[i:i+window_size]
        sig = ta.extract_signal(curr.Open.as_matrix(), curr.High.as_matrix(), curr.Low.as_matrix(), curr.Close.as_matrix())
        date = ohlc.index[i+window_size-1]
        all_signals[date] = sig
    all_signals = pd.Series(all_signals, name='signals')
    return all_signals


def get_n_day_trend(ohlc, n=3, threshold=0.01):
    """
    Trend calculated based on c_{t+n} / c_{t} at threshold
    """
    current = ohlc.Close
    future = current.shift(-n)
    bull = (future / current) > (1+threshold)
    bear = ((future / current) < (1-threshold)) * -1
    no_trend = pd.Series(np.zeros(len(current)),index=current.index)
    trend = bull + bear + no_trend
    return pd.Series(trend[:-n], name='n_trend')

def get_window_trend(ohlc, n=3, threshold=0.01):
    """
    Trend calculated based on c_{t} / c_{t-n} at threshold
    """
    n = n-1 #inclusive of time t=3 for window size n=3
    current = ohlc.Close
    prev = current.shift(n)
    bull = (current / prev) > (1+threshold)
    bear = ((current / prev) < (1-threshold)) * -1
    no_trend = pd.Series(np.zeros(len(current)),index=current.index)
    trend = bull + bear + no_trend
    return pd.Series(trend[n:], name='n_trend')

def bucket_trend_to_cluster(ohlc, K):
    #bucket trend to cluster
    overlaps = {}
    for i in range(K):
        overlaps[i] = []
    for i, j in zip(ohlc.Cluster.astype(int), ohlc.n_trend.astype(int)):
        overlaps[i].append(j)
    return overlaps

def bucket_trend_to_signal(ohlc):
    #bucket trend to cluster
    overlaps = {}
    for i, j in zip(ohlc.signals, ohlc.n_trend.astype(int)):
        if i not in overlaps.keys():
            overlaps[i] = []
        overlaps[i].append(j)
    return overlaps

def count_signal_type(overlaps, cluster):
    """
    Counts the distribution of a signal (trend, candle) in a cluster
    Params:
        overlaps: dict. {cluster: [signal1, signal2, signal1,...]}
    Return:
        dist: pd Series distributino of signal in cluster
    """
    curr = Counter(overlaps[cluster])
    for key in curr.keys():
        curr[key] /= float(len(overlaps[cluster]))
    return pd.Series(curr, name='cluster_'+str(cluster))

def count_distribution(data):
    data = pd.Series(Counter(data))
    data = data.div(data.sum())
    return data.to_dict()

def trend_cluster_distribution(trend_cluster, K):
    overlaps = bucket_trend_to_cluster(trend_cluster, K)
    return pd.concat([count_signal_type(overlaps, i) for i in overlaps.keys()],axis=1).fillna(0)