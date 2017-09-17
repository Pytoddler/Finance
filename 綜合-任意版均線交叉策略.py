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
import os

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

################ 策略執行 ################

tmp_bc = 0
Buy_count = [] #初始買進次數

tmp_sc = 0
Sell_count = [] #初始賣出次數
Trade_count = [] #總交易次數
#trade_fee = 0.01#每次交易都滑價+手續費

##表格上每個時間點的變數紀錄

tmp_p = 0 #儲存目前的state紀錄，表示0不持有,1持有
Position = [] #是否持有股票各個時間點加總的list

Trading_act = [] #紀錄所有交易是買還是賣
Trading_price = [] #紀錄所有交易價格
Trading_date = [] #紀錄所有交易時間

tmp_bt_pct_change = 0
Bt_pct_change = [] #記錄每日策略的波動率
Bt_pct_change.append(0) #第一天是0

tmp_bt_logcum_pct = 0 #儲存每個時間點「交易策略」的Log_cum_pct_change
Bt_logcum_pct = [] #初始100%

#儲存每個時間點的Max_drawdown_pct，初始數值0
tmp_bt_mdd_pct = 0 #儲存每個時間點的Max_drawdown_pct
Bt_mdd_pct = []
tmp_m_mdd_pct = 0
M_mdd_pct = []

tmp_m_logcum_pct = 0 
M_cum_pct = [] #儲存每個時間點的市場總%變化率
M_ret = [] #市場總報酬率 = 市場總%變化率 - 100%

Bt_cum_pct = [] #儲存每個時間點的策略總%變化率
Bt_ret = [] #策略總報酬率 = 策略總%變化率 - 100%

datenum = len(df.index.tolist())
Date = df.index.tolist()
Price = df['Adj Close'].tolist()
short_MA = df[short+'MA'].tolist()
long_MA = df[long+'MA'].tolist()
Vol = df['Volume'].tolist()
Tg_log_pct_change = df['Tg_log_pct_change'].tolist() #市場每天的Log%變化


#策略執行、記錄
#使用一個for迴圈跑完全部的回測
for i in range(datenum): #把所有日期run一遍

    #每天記錄: Position, Buy count, Sell_count, Trade_count 
    #每天記錄: Bt_%_change, Bt_logcum%, M_cum%, M_ret, Bt_cum%, Bt_ret, Bt_mdd%, M_mdd%, 
    #Trading有交易才記錄: Trading_act(Buy或者Sell), Trading_price, Trading_date

    #這四個狀態是互斥的，因此用if跟elif撰寫

    if short_MA[i] > long_MA[i] and tmp_p <= 0: #有進出場，要記錄
        tmp_p = tmp_p + 1
        Position.append(tmp_p)

        tmp_bc += 1 #只有買進變化
        Buy_count.append(tmp_bc)

        tmp_sc = tmp_sc
        Sell_count.append(tmp_sc)

        #交易發生:紀錄在新的表格!!
        print(str(Date[i])+': Buy - '+str(Price[i]))
        Trading_act.append('Buy')
        Trading_date.append(Date[i])
        Trading_price.append(Price[i])
        
    elif short_MA[i] > long_MA[i] and tmp_p > 0: #繼續持有
        tmp_p = tmp_p
        Position.append(tmp_p)

        tmp_bc = tmp_bc
        Buy_count.append(tmp_bc)

        tmp_sc = tmp_sc
        Sell_count.append(tmp_sc)
        
    elif short_MA[i] < long_MA[i] and tmp_p <= 0: #繼續空手
        tmp_p = tmp_p
        Position.append(tmp_p)

        tmp_bc = tmp_bc
        Buy_count.append(tmp_bc)

        tmp_sc = tmp_sc
        Sell_count.append(tmp_sc)

    elif short_MA[i] < long_MA[i] and tmp_p > 0: #有進出場，要記錄
        tmp_p = tmp_p - 1
        Position.append(tmp_p)

        tmp_bc = tmp_bc
        Buy_count.append(tmp_bc)

        tmp_sc += 1 #只有賣出變化
        Sell_count.append(tmp_sc)

        #交易發生:紀錄在新的表格!!
        print( str(Date[i])+': Sell - '+str(Price[i]) )
        Trading_act.append('Sell')
        Trading_date.append(Date[i])
        Trading_price.append(Price[i])

    Trade_count.append( Buy_count[i] + Sell_count[i] )


    #記錄策略log總%報酬率
    #利用tmp>0代表持有股票時，cum報酬率每回合都要加上一次當日log報酬率
    if tmp_p > 0: #只計算做多，因為只有做多
        tmp_bt_logcum_pct += Tg_log_pct_change[i]
        Bt_logcum_pct.append(tmp_bt_logcum_pct)
    else: #不做空，累積報酬率不變化，pass
        tmp_bt_logcum_pct = tmp_bt_logcum_pct
        Bt_logcum_pct.append(tmp_bt_logcum_pct)


    ##記錄總%變化、報酬率
    #市場總%變化、報酬率
    tmp_m_logcum_pct += Tg_log_pct_change[i]

    tmp4 = np.exp(tmp_m_logcum_pct)*100
    M_cum_pct.append(tmp4)

    tmp4_1 = np.exp(tmp_m_logcum_pct)*100 - 100
    M_ret.append(tmp4_1)
    
    #策略總%變化、報酬率
    tmp5 = np.exp( Bt_logcum_pct[i] + Trade_count[i]*np.log(1-float(trade_fee)) )*100
    Bt_cum_pct.append(tmp5)
    
    tmp5_1 = np.exp( Bt_logcum_pct[i] + Trade_count[i]*np.log(1-float(trade_fee)) )*100 - 100
    Bt_ret.append(tmp5_1)

    #市場每日%變化，作為波動率計算: 已經在第一部分資料處理就算完畢了
    
    #策略每日%變化，作為波動率計算
    if i >= 1 :
        tmp_bt_pct_change = (Bt_cum_pct[i]/Bt_cum_pct[i-1])*100 - 100
        Bt_pct_change.append(tmp_bt_pct_change)

    ##Max_Drawdown
    #市場最大虧損%數，若<1則帶入數值出來  
    tmp_m_mdd_pct = (M_cum_pct[i]-max(M_cum_pct))/max(M_cum_pct) *100
    M_mdd_pct.append(tmp_m_mdd_pct)
    

    #策略最大虧損%數: 找到min(Backtest_return)，若<1則帶入數值出來       
    tmp_bt_mdd_pct = (Bt_cum_pct[i]-max(Bt_cum_pct))/max(Bt_cum_pct) *100
    Bt_mdd_pct.append(tmp_bt_mdd_pct)

print('\n')
#統計靜態結果 
m_volatility = np.std(M_cum_pct)
bt_volatility = np.std(Bt_cum_pct)
m_sharpe_ratio = M_ret[-1] / m_volatility
bt_sharpe_ratio = Bt_ret[-1] / bt_volatility

print('\n')
        
df['Position'] = Position
df['M_cum_pct'] = M_cum_pct
df['M_ret'] = M_ret
df['Bt_cum_pct'] = Bt_cum_pct
df['Bt_ret'] = Bt_ret
df['Bt_volatility'] = Bt_pct_change
df['Bt_logcum_pct'] = Bt_logcum_pct
df['M_mdd_pct'] = M_mdd_pct
df['Bt_mdd_pct'] = Bt_mdd_pct

'''
print('##### Backtesting strategy Summary #####')
print('target_ID: '+stock_ID)
print('backtesting strategy: '+method_name)
print('total backtesting days: '+str(datenum))
print('trade_count: '+str(Trade_count[-1]))#操作次數
print('trade_fee: '+str(float(trade_fee*100))+'%')
print('M_cum_pct: '+ str(M_cum_pct[-1]) + '%, M_ret: '+str(M_ret[-1])+'%')
print('Bt_cum_pct: ' + str(Bt_cum_pct[-1]) + '%, Bt_ret: '+str(Bt_ret[-1])+'%')
print('M_volatility:'+str(m_volatility)
print('Bt_volatility:'+str(bt_volatility)
print('M_mdd_pct: ' + str(min(M_mdd_pct))+'%')
print('Bt_mdd_pct: ' + str(min(Bt_mdd_pct))+'%')
print('m_sharpe_ratio: '+str(m_sharpe_ratio))
print('bt_sharpe_ratio: '+str(bt_sharpe_ratio))
print('current position: ' + str(Position[-1]))
'''

Backtesting_strategy_Summary = '##### Backtesting strategy Summary #####'+'\n'+'target_ID: '+stock_ID+'\n'+'backtesting strategy: '+method_name+'\n'+'total backtesting days: '+str(datenum)+'\n'+'trade_count: '+str(Trade_count[-1])+'\n'+'trade_fee: '+str(float(trade_fee)*100)+'%'+'\n'+'M_cum%: '+ str(M_cum_pct[-1]) + '%, M_ret: '+str(M_ret[-1])+'%'+'\n'+'Bt_cum%: ' + str(Bt_cum_pct[-1]) + '%, Bt_ret: '+str(Bt_ret[-1])+'%'+'\n'+'M_volatility%: '+str(m_volatility)+'%'+'\n'+'Bt_volatility%: '+str(bt_volatility)+'%'+'\n'+'M_mdd%: ' + str(min(M_mdd_pct))+'%'+'\n'+'Bt_mdd%: ' + str(min(Bt_mdd_pct))+'%'+'\n'+'m_sharpe_ratio: '+str(m_sharpe_ratio)+'\n'+'bt_sharpe_ratio: '+str(bt_sharpe_ratio)+'\n'+'current position: ' + str(Position[-1])
filename_summary = '【Backtest】summary_'+stock_ID+' '+method_name+'.txt'
filename_summary = os.path.join(sav_path, filename_summary)

with open(filename_summary,'w') as f:
    f.write(Backtesting_strategy_Summary)
    f.close()

Trading_record = {'Trading_date': Trading_date,
                  'Trading_act': Trading_act,
                  'Trading_price': Trading_price}
columns = ['Trading_date','Trading_act', 'Trading_price']
df2 = pd.DataFrame(data = Trading_record, columns= columns)


filename_Record = '【Backtest】Record_'+stock_ID+' '+method_name+'.csv'
filename_Record = os.path.join(sav_path, filename_Record)
df2.to_csv(filename_Record, index_label='Trading_count')

filename_Fulldata = '【Backtest】Fulldata_'+stock_ID+' '+method_name+'.csv'
filename_Fulldata = os.path.join(sav_path, filename_Fulldata)
df.to_csv(filename_Fulldata, index_label='Date')

print('\n')
print('backtest record datahead')
print(df2.head()) #測試用
              
print('\n')
print('backtest total datahead')
print(df.head()) #測試用

###
#回測第三步驟，由已經收集到的資料進行分析和統計結果
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
ax3.legend(loc='best', prop={'size': 8})
ax4.legend(loc='best', prop={'size': 6})

#png存檔，展示
filename_png = '【Backtest】summary_'+stock_ID+' '+method_name+'.png'
filename_png = os.path.join(sav_path, filename_png)
fig.savefig(filename_png) 
plt.show()

#也一起讀出統計資料
with open(filename_summary,'r') as f:
    for line in f:
        print (line)
