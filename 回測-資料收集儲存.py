# 回測第一步驟，收集資料，並且用CSV檔案儲存

import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
from matplotlib.finance import candlestick_ohlc
import numpy as np
from collections import Counter
import pickle


#回測第一步驟，收集資料，並且用csv檔案儲存

#把想要的股票、日期區間直接變成FataFrame
def stock_df(stock_ID, start, end):
    df = pdr.DataReader(stock_ID,'yahoo', start, end)
    df.dropna(inplace=True)
    return df

#定義股票ID，想抓的start, end時間點
stock_ID = input('stock_ID: ')
short = input('short interval (days): ')
long = input('long interval (days): ')
trade_fee = input('trade_fee，ex.0.01，不要輸入百分比: ')

method_name = short+'MA_'+long+'MA_cross_long'

start = dt.datetime(2001,1,1)
end = date.today() #直接回溯到今天


#以標的名稱為命名，創建該標的的檔案資料夾
def create_path(sav_path):
    try:
        os.makedirs(sav_path)
    except OSError:
        if os.path.exists(sav_path):
            # We are nearly safe
            pass
        else:
            # There was an error on creation, so make sure we know about it
            raise

sav_path = 'C:\\Documents and Settings\\Administrator\\My Documents\\Py34\\Finance\\股票回測\\'+stock_ID
create_path(sav_path)


##如果有存過檔案就不用再跑一次，直接讀取
df = stock_df(stock_ID, start, end)
print(df.head())


#儲存yahoo data上面最原始的資料
filename_yr = stock_ID+'_yh.csv'
filename_yr = os.path.join(sav_path, filename_yr)
df.to_csv(filename_yr, index_label='Date')
df = pd.read_csv(filename_yr, parse_dates=True, index_col=0) #沒有parse_dates就會多一整欄位！！


###########################################
#到目前為止整個df的架構都可以自由操作了

#短日線、長日線
df[short+'MA'] = pd.Series.rolling(df['Adj Close'], window=int(short)).mean()
df[long+'MA'] = pd.Series.rolling(df['Adj Close'], window=int(long)).mean()

#標的每日%變動數和log(%變動數) (今天-昨天/昨天)+1
df['Tg_pct_change'] = 1 + (df['Adj Close'] - df['Adj Close'].shift(1) ) / df['Adj Close'].shift(1)
#標的log處理過的每日%變動數，使用log總報酬可以直接用加的
df['Tg_log_pct_change'] = np.log(df['Tg_pct_change'])


#統整所有資料準備回測！！
df.dropna(inplace=True) #把head沒有ma的na部分去掉避免麻煩
print(df.head())

#儲存處理過準備跑回測的資料
filename_prodata = stock_ID+'_MA_%_波動.csv'
filename_prodata= os.path.join(sav_path, filename_prodata)
df.to_csv(filename_prodata, index_label='Date')

#回測第二步驟，由已經收集到的資料進行分析和統計結果
#利用已經儲存的資料
df = pd.read_csv(filename_prodata, parse_dates=True, index_col=0) #沒有parse_dates就會多一整欄位！！
print('loadcsv2\n')
print(df.head())
