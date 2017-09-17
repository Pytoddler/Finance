#回測第二步驟，由已經收集到的資料進行分析和統計結果

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
import matplotlib.ticker as mticker

#把想要的股票、日期區間直接變成FataFrame
def stock_df(stock_ID, start, end):
    df = pdr.DataReader(stock_ID,'yahoo', start, end)
    df.dropna(inplace=True)
    return df

#定義股票ID，想抓的start, end時間點
stock_ID = '^TWII'
method_name = '20MA_60MA_cross_long'

#利用已經儲存的資料
df = pd.read_csv(stock_ID+'2.csv', parse_dates=True, index_col=0) #沒有parse_dates就會多一整欄位！！

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
