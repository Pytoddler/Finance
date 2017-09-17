#回測第三步驟，由已經收集到的資料進行分析和統計結果

import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import pandas_datareader.data as pdr
import fix_yahoo_finance as yf
from matplotlib.finance import candlestick_ohlc
import numpy as np
from collections import Counter
import pickle
import matplotlib.ticker as mticker

#定義股票ID，想抓的start, end時間點
stock_ID = '^TWII'
method_name = '20MA_60MA_cross_long'

###
#回測第三步驟，由已經收集到的資料進行分析和統計結果
sav_path = 'C:\\Documents and Settings\\Administrator\\My Documents\\Py34\\Finance\\股票回測\\'+stock_ID

filename_summary = '【Backtest】summary_'+stock_ID+' '+method_name+'.txt'
filename_summary = os.path.join(sav_path, filename_summary)

filename_Record = '【Backtest】Record_'+stock_ID+' '+method_name+'.csv'
filename_Record = os.path.join(sav_path, filename_Record)

filename_Fulldata = '【Backtest】Fulldata_'+stock_ID+' '+method_name+'.csv'
filename_Fulldata = os.path.join(sav_path, filename_Fulldata)

df = pd.read_csv(filename_Fulldata, parse_dates=True, index_col=0)
df2 = pd.read_csv(filename_Record, parse_dates=True, index_col=1)

#print(df.head()) #測試用
#print(df2.head()) #測試用

############## 以下資料視覺化 ##############
#圖一: OHLC K線圖 + 20MA,60MA 雙均線
#圖二: Vol圖
#圖三: Market_cum_pct_change 和 Backtest_cum_pct_change作圖
#圖四: Max_drawdown圖
#x軸: 對齊時間

#先寫好大的figure
fig = plt.figure()
ax1 = plt.subplot2grid((11,8),(0,0), rowspan=7, colspan=8)
ax1.set_title('Backtest: '+stock_ID+' '+method_name)
ax1.set_ylabel('Adj price')
ax1.patch.set_facecolor('lightyellow') #改變背景顏色

ax2 = plt.subplot2grid((11,8),(7,0), rowspan=1, colspan=8, sharex=ax1)
ax2.set_ylabel('Position')
ax2.yaxis.set_major_locator(mticker.MaxNLocator(1))
ax2.patch.set_facecolor('lightyellow') #改變背景顏色

ax3 = plt.subplot2grid((11,8),(8,0), rowspan=2, colspan=8, sharex=ax1)
ax3.set_ylabel('Asset%')
ax3.yaxis.set_major_locator(mticker.MaxNLocator(6))
ax3.patch.set_facecolor('lightyellow') #改變背景顏色

ax4 = plt.subplot2grid((11,8),(10,0), rowspan=1, colspan=8, sharex=ax1)
ax4.set_ylabel('Max_DD%')
ax4.set_xlabel('Date')
ax4.yaxis.set_major_locator(mticker.MaxNLocator(3))
ax4.patch.set_facecolor('lightyellow') #改變背景顏色

#再聚焦每一張子圖
ax1.plot(df.index, df['Adj Close'], linewidth=1.2, color='orange', label='Adj Close')
#df_ohlc = pd.DataFrame([df['Open'],df['High'],df['Low'],df['Adj Close']])
#candlestick_ohlc(ax1, df_ohlc.values, width=1, colorup='r', colordown='g') 
ax1.plot(df.index, df[short+'MA'], color='c', label=short+'MA')
ax1.plot(df.index, df[long+'MA'], color='b', label=long+'MA')
ax1.fill_between(df.index, df[short+'MA'], df[long+'MA'], where = df[short+'MA']>df[long+'MA'], color= 'red', alpha=1)
ax1.fill_between(df.index, df[short+'MA'], df[long+'MA'], where = df[short+'MA']<df[long+'MA'], color= 'chartreuse', alpha=1)
ax1.scatter(df2.index, df2['Trading_price'], c='firebrick', alpha=1)
bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
ax1.annotate(str(df['Adj Close'][-1]), (df.index[-1], df['Adj Close'][-1]), xytext=(df.index[-1], df['Adj Close'][-1]), bbox=bbox_props)

#ax2.fill_between(df.index, df['Volume'], color='dodgerblue', label='Vol')
ax2.fill_between(df.index, df['Position'], color='dodgerblue', label='Position') #第二張畫部位狀態比較好

ax3.plot(df.index, df['Bt_cum_pct'], color='r', label='Backtest % (Trade_fee=1%)')
ax3.plot(df.index, df['M_cum_pct'], color='lime', label='Market %')
ax3.axhline(y=100.00, color='navy', linestyle='--')
bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
ax3.annotate(str(df['Bt_cum_pct'][-1]), (df.index[-1], df['Bt_cum_pct'][-1]), xytext=(df.index[-1], df['Bt_cum_pct'][-1]), bbox=bbox_props)
ax3.annotate(str(df['M_cum_pct'][-1]), (df.index[-1], df['M_cum_pct'][-1]), xytext=(df.index[-1], df['M_cum_pct'][-1]), bbox=bbox_props)
#ax3.annotate('100.00', (df.index[-1], 100), xytext=(df.index[-1], 100), bbox=bbox_props)

ax4.plot(df.index, df['Bt_mdd_pct'], color='red', label='Backtest %')
ax4.plot(df.index, df['M_mdd_pct'], color='lime', label='Market %')
bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
ax4.annotate(str(df['Bt_mdd_pct'][-1]), (df.index[-1], df['Bt_mdd_pct'][-1]), xytext=(df.index[-1], df['Bt_mdd_pct'][-1]), bbox=bbox_props)
ax4.annotate(str(df['M_mdd_pct'][-1]), (df.index[-1], df['M_mdd_pct'][-1]), xytext=(df.index[-1], df['M_mdd_pct'][-1]), bbox=bbox_props)

#把legend附上去
ax1.legend(loc=4)
ax2.legend(loc='best', prop={'size': 10})
ax3.legend(loc=2, prop={'size': 8})
ax4.legend(loc=10, prop={'size': 6})

#png存檔，展示
filename_png = '【Backtest】summary_'+stock_ID+' '+method_name+'.png'
filename_png = os.path.join(sav_path, filename_png)
fig.savefig(filename_png) 
plt.show()

#也一起讀出統計資料
with open(filename_summary,'r') as f:
    for line in f:
        print (line)
