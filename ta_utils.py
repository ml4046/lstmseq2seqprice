"""
Utilities for technical indicators 
"""
import numpy as np
from pandas import ewma
import pandas as pd
######################Major signals######################
def bull_engulf(Open, high, low, close, t=4):
    """
    Identifies if prices is a Bullish Engulfing Pattern of not
    
    Param:
        Open: array of open prices (5-day)
        high: array of high prices (5-day)
        low: array of low prices (5-day)
        close: array of close prices (5-day)
        t: int num. day -1 (5-1=4)
    Return:
        status: boolean true if it is the pattern
    """
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
        
    if (Open[t] < close[t-1] and 
        close[t] > Open[t-1] and 
        Open[t] < close[t] and 
        Open[t-1] > close[t-1] and
        (Open[t-2]>close[t-2] and Open[t-3]>close[t-3] and Open[t-4]>close[t-4]) and
        (close[t-2]<close[t-3]<close[t-4])):
        return True
    
    return False

def bear_engulf(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
    if (Open[t] > close[t-1] and
        close[t] < Open[t-1] and
        Open[t] > close[t] and
        Open[t-1] < close[t-1] and
       (Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and Open[t-4]<close[t-4]) and
       (close[t-2]>close[t-3]>close[t-4])):
        return True
    
    return False

def hammer(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
    if Open[t] > close[t]: #whitebody
        if (high[t]==close[t] and 
            (Open[t]-low[t]) > 2*(close[t]-Open[t]) and
           (Open[t-2]>close[t-2] and Open[t-3]>close[t-3] and Open[t-4]>close[t-4]) and
           (close[t-2]<close[t-3]<close[t-4])):
            return True
            
    elif Open[t] < close[t]: #blackbody
        if (high[t]==Open[t] and
            (close[t]-low[t]) > 2*(Open[t]-close[t]) and
           (Open[t-2]>close[t-2] and Open[t-3]>close[t-3] and Open[t-4]>close[t-4]) and
           (close[t-2]<close[t-3]<close[t-4])):
            return True
    return False

def hanging_man(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
    if Open[t] > close[t]: #whitebody
        if (high[t]==close[t] and 
            (Open[t]-low[t]) > 2*(close[t]-Open[t]) and
           (Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and Open[t-4]<close[t-4]) and
           (close[t-2]>close[t-3]>close[t-4])):
            return True
            
    elif Open[t] < close[t]: #blackbody
        if (high[t]==Open[t] and
            (close[t]-low[t]) > 2*(Open[t]-close[t]) and
           (Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and Open[t-4]<close[t-4]) and
           (close[t-2]>close[t-3]>close[t-4])):
            return True
    return False

def piercing_line(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
        
    if (Open[t-1]>close[t-1] and
        Open[t]<low[t-1] and
        close[t] > (close[t-1]+0.5*(Open[t-1]-close[t-1])) and
        (Open[t-2]>close[t-2] and Open[t-3]>close[t-3] and Open[t-4]>close[t-4]) and
        (close[t-2]<close[t-3]<close[t-4])):
        return True
    return False

def dark_cloud_cover(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
        
    if (Open[t-1]<close[t-1] and
        Open[t]>high[t-1] and
        close[t] < (close[t-1]-0.5*(close[t-1]-Open[t-1])) and
       (Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and Open[t-4]<close[t-4]) and
       (close[t-2]>close[t-3]>close[t-4])):
        return True
    return False


def bull_harami(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
        
    if (close[t-1]<Open[t]<close[t]<Open[t-1] and
        (Open[t-2]>close[t-2] and Open[t-3]>close[t-3] and Open[t-4]>close[t-4]) and
        (close[t-2]<close[t-3]<close[t-4])):
        return True
    return False

def bear_harami(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices are not length 5')
        
    if (Open[t-1]<close[t]<Open[t]<close[t-1] and
       (Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and Open[t-4]<close[t-4]) and
       (close[t-2]>close[t-3]>close[t-4])):
        return True
    return False

def morning_star(Open, high, low, close, t=5):
    if len(Open) < 6:
        raise AttributeError('Prices less than 6')
        
    if (max(close[t-1], Open[t-1])<Open[t]<close[t-2]<
        close[t-2]+0.5*(Open[t-2]-close[t-2]) < close[t] < Open[t-2] and
        (Open[t-3]>close[t-3] and Open[t-4]>close[t-4] and Open[t-5]>close[t-5]) and
        (close[t-3]<close[t-4]<close[t-5])):
        return True
    return False

def evening_star(Open, high, low, close, t=5):
    if len(Open) < 6:
        raise AttributeError('Prices less than 6')

    if (max(close[t-1], Open[t-1])>Open[t]>close[t-2]>
        close[t-2]-0.5*(close[t-2]-Open[t-2]) > close[t] > Open[t-2] and
        (Open[t-3]<close[t-3] and Open[t-4]<close[t-4] and Open[t-5]<close[t-5]) and
        (close[t-3]>close[t-4]>close[t-5])):
        return True
    return False
        
def bull_kicker(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
        
    if (close[t]>Open[t]==Open[t-1]>close[t-1] and
        Open[t-2]>close[t-2] and Open[t-3] > close[t-3] and
        close[t-2] < close[t-3]):
        return True
    return False
        
    
def bear_kicker(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (close[t-1]>Open[t-1]==Open[t]>close[t] and
        Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and
        close[t-2]>close[t-3]):
        return True
    return False

def shooting_star(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (high[t]>max(Open[t],close[t]) and
        high[t]-max(Open[t],close[t])>= 2*abs(Open[t]-close[t]) and
        Open[t-1]<close[t-1] and Open[t-2]<close[t-2] and Open[t-3]<close[t-3] and
        close[t-1]>close[t-2]>close[t-3] and
        low[t]==min(Open[t],close[t])):
        return True
    return False


def inverted_hammer(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
        
    if (high[t]-max(Open[t],close[t])>=2*abs(Open[t]-close[t]) and
        low[t]==min(Open[t],close[t]) and
        close[t-1]>max(Open[t],close[t]) and
        Open[t-1]>close[t-1] and Open[t-2]>close[t-2] and Open[t-3]>close[t-3] and
        close[t-1]<close[t-2]<close[t-3]):
        return True
    return False

######################Minor signals######################
def three_black_crows(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices less than 5')
    if (Open[t]>close[t] and Open[t-1]>close[t-1] and Open[t-2]>close[t-2] and
        Open[t-2]>Open[t-1]>close[t-2] and
        Open[t-1]>Open[t]>close[t-1] and
        Open[t-3]<close[t-3] and Open[t-4]<close[t-4]):
        return True
    return False

def three_white_soldiers(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices less than 5')
    if (Open[t]<close[t] and Open[t-1]<close[t-1] and Open[t-2]<close[t-2] and
        Open[t-2]<Open[t-1]<close[t-2] and
        Open[t-1]<Open[t]<close[t-1] and
        Open[t-3]>close[t-3] and Open[t-4]>close[t-4]):
        return True
    return False

def upside_gap_2crows(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices less than 5')
    if (Open[t]>Open[t-1]>close[t-1]>close[t]>close[t-2]>Open[t-2] and
        close[t-2]>close[t-3]>close[t-4]):
        return True
    return False

def stick_sandwich(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices less than 5')
    if (Open[t]>close[t-1]>Open[t-2]>Open[t-1]>close[t-2]==close[t] and
        close[t-2]<close[t-3]<close[t-4]):
        return True
    return False

def bull_meeting_lines(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (Open[t]<close[t]==close[t-1]<Open[t-1] and
        close[t-1]<close[t-2]<close[t-3]):
        return True
    return False

def bear_meeting_lines(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (Open[t]>close[t]==close[t-1]>Open[t-1] and
        close[t-1]>close[t-2]>close[t-3]):
        return True
    return False

def deliberation(Open, high, low, close, t=4):
    if len(Open) < 5:
        raise AttributeError('Prices less than 5')
    if (close[t]>Open[t] and close[t-1]>Open[t-1] and close[t-2]>Open[t-2] and
        close[t]>close[t-1]>close[t-2] and
        close[t-1]-Open[t-1] > 2*(close[t]-Open[t]) and close[t-2]-Open[t-2] > 2*(close[t]-Open[t]) and
        abs((close[t-1]-Open[t-1]) - (close[t-2]-Open[t-1])) < 1e-3 and #sort of similar
        close[t-2]>close[t-3]>close[t-4]):
        return True
    return False

def homing_pigeon(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (Open[t-1]>Open[t]>close[t]>close[t-1] and
        close[t-1]<close[t-2]<close[t-3] ):
        return True
    return False

def matching_low(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (Open[t-1]>Open[t]>close[t]==close[t-1] and
        close[t-1]<close[t-2]<close[t-3]):
        return True
    return False

def matching_high(Open, high, low, close, t=3):
    if len(Open) < 4:
        raise AttributeError('Prices less than 4')
    if (Open[t-1]<Open[t]<close[t]==close[t-1] and
        close[t-1]>close[t-2]>close[t-3]):
        return True
    return False

def extract_signal(Open, high, low, close):
    """
    Extracts a defined candlestick signal, if no signal found 'no_signal' is returned
    
    Param:
        Open: array of open prices (5-day)
        high: array of high prices (5-day)
        low: array of low prices (5-day)
        close: array of close prices (5-day)
    Return:
        signal: str name of signal
    """
    if bull_engulf(Open, high, low, close):
        return 'bull_engulf'
    elif bear_engulf(Open, high, low, close):
        return 'bear_engulf'
    elif hammer(Open, high, low, close):
        return 'hammer'
    elif hanging_man(Open, high, low, close):
        return 'hanging_man'
    elif piercing_line(Open, high, low, close):
        return 'piercing_man'
    elif dark_cloud_cover(Open, high, low, close):
        return 'dark_cloud_cover'
    elif bull_harami(Open, high, low, close):
        return 'bull_harami'
    elif bear_harami(Open, high, low, close):
        return 'bear_harami'
    elif morning_star(Open, high, low, close):
        return 'morning_star'
    elif evening_star(Open, high, low, close):
        return 'evening_star'
    elif bull_kicker(Open, high, low, close):
        return 'bull_kicker'
    elif bear_kicker(Open, high, low, close):
        return 'bear_kicker'
    elif shooting_star(Open, high, low, close):
        return 'shooting_star'
    elif inverted_hammer(Open, high, low, close):
        return 'inverted_hammer'
    ###minor signal###
    elif three_black_crows(Open, high, low, close):
        return 'three_black_crows'
    elif three_white_soldiers(Open, high, low, close):
        return 'three_white_soldiers'
    elif upside_gap_2crows(Open, high, low, close):
        return 'upside_gap_2crows'
    elif stick_sandwich(Open, high, low, close):
        return 'stick_sandwich'
    elif bull_meeting_lines(Open, high, low, close):
        return 'bull_meeting_lines'
    elif bear_meeting_lines(Open, high, low, close):
        return 'bear_meeting_lines'
    elif deliberation(Open, high, low, close):
        return 'deliberation'
    elif homing_pigeon(Open, high, low, close):
        return 'homing_pigeon'
    elif matching_low(Open, high, low, close):
        return 'matching_low'
    elif matching_high(Open, high, low, close):
        return 'matching_high'
    return 'no_signal'

######################TA indicator utils######################

############################################
##Volatility
############################################
def bollinger(close, window_size=20, K=2):
    """
    Returns tuple of bollinger band vol. measure
    (MA + K*sig), (MA - K*sig)
    """
    ma_values = ma(close, window_size)
    sig = ma_values.std()
    return (ma_values + K*sig, ma_values - K*sig)

def perBollinger(close, window_size=20, K=2):
    """
    %b indicator measures the width of upper/lower bands over time
    """
    uband, lband = bollinger(close, window_size=window_size, K=K) 
    return (close[window_size-1:]-lband)/(uband-lband)

def ATR(high, low, close, window=14):
    """
    Price volatility index
    """
    #prev
    high_prev = high[:len(high)-1]
    low_prev = low[:len(low)-1]
    close_prev = close[:len(close)-1]
    #curr
    high = high[1:]
    low = low[1:]
    close = close[1:]
    TR = np.maximum((high-low), abs(high-close_prev), abs(low-close_prev))
    com = (window-1)/2.0
    return ewma(pd.Series(TR), com=com).as_matrix()

############################################
##Trend
############################################

def ma(interval, window_size):
    """
    Finds window_size ma for interval (shoutout to IEOR8100)
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

def ema(values, window):
    """ 
    Numpy implementation of EMA
    https://pythonprogramming.net/advanced-matplotlib-graphing-charting-tutorial/
    """
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def MACD(close, short=12, long_=26, signal=9):
    short_ema = ema(close, short)
    long_ema = ema(close, long_)
    sig_ema = ema(close, signal)
    macd = short_ema[len(short_ema)-len(long_ema):] - long_ema
    return macd - sig_ema[len(sig_ema)-len(long_ema):]

def CCI(high, low, close, window=20):
    """
    Commodity Channel Index (identify cyclical trend)
    """
    def moving_mad(price, averages, window=window):
        mads = []
        for i in range(len(averages)):
            mads.append(np.mean(abs(price[i:i+window]-averages[i])))
        return np.array(mads)
    p_mid = (high + low + close) / 3.0
    p_ma = ma(p_mid, window)
    p_mad = moving_mad(p_mid, p_ma)
    p_diff = (1/0.015) * ((p_mid[window-1:] - p_ma) / p_mad)
    return p_diff

def rate_of_change(close, n=1):
    """
    Rate of change in price for n days ago
    """
    return (close[n:]-close[:len(close)-n])/close[:len(close)-n]

############################################
##Volume
############################################

def ADI(high, low, close, volume):
    """
    accumulation/distribution index
    
    replaces inf with max/min
    """
    clv = ((close-low) - (high-close))/(high-close)
    #clv[np.isposinf(clv)] = max(clv[~np.isposinf(clv)])
    #clv[np.isneginf(clv)] = min(clv[~np.isneginf(clv)])
    clv[np.isinf(clv)] = 0
    adi = (clv[:-1] + volume[:-1]) * clv[1:]
    #adi[np.isposinf(adi)] = max(adi[~np.isposinf(adi)])
    #adi[np.isneginf(adi)] = min(adi[~np.isneginf(adi)])    
    return adi

def OBV(close, volume):
    """
    On-Balance Volume
    """
    obv = np.zeros(len(close))
    obv[0] = volume[0]
    greater_index = np.where(close[1:]>close[:-1])[0]
    less_index = np.where(close[1:]<close[:-1])[0]
    equal_index = np.where(close[1:]==close[:-1])[0]
    obv[greater_index+1] = volume[greater_index]
    obv[less_index+1] = -volume[less_index]
    obv[equal_index+1] = 0
    return obv[:-1] + obv[1:]

def CMF(high, low, close, volume, n=20, fillna=False):
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
    Returns:
        pandas.Series: New feature generated.
    https://github.com/bukosabino/ta/blob/master/ta/volume.py
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0) # float division by zero
    mfv *= volume
    cmf = mfv.rolling(n).sum() / volume.rolling(n).sum()
    if fillna:
        cmf = cmf.fillna(0)
        
    return cmf.dropna().reset_index(drop=True)
############################################
##Momentum
############################################
def RSI(close, n=14, fillna=False):
    """Relative Strength Index (RSI)
    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.
    https://www.investopedia.com/terms/r/rsi.asp
    Args:
        close(np array): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
        
        https://github.com/bukosabino/ta/blob/master/ta/momentum.py
    """
    close = pd.Series(close)
    diff = close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = up.ewm(n).mean()
    emadn = dn.ewm(n).mean()

    rsi = 100 * emaup/(emaup + emadn)
    if fillna:
        rsi = rsi.fillna(50)
    return rsi.dropna().reset_index(drop=True)

def SMI(high, low, close, n=14):
    """
    Stochastic Momentum Index
    """
    high_max = np.array([max(high[i:i+14]) for i in range(len(high)-n)])
    low_min = np.array([min(low[i:i+14]) for i in range(len(low)-n)])
    midpoint = (high_max + low_min) / 2.0
    diff = close[n:] - midpoint
    diff_ema = ema(ema(diff,3),3)
    high_low_range_ema = ema(ema(high_max - low_min,3),3)
    return diff_ema/high_low_range_ema

def WVAD(Open, high, low, close, volume):
    """
    Volume weighted price momentum index
    """
    return ((close - Open)/(high - low)) * volume