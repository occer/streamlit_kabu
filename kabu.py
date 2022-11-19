import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# YahooDailyReader のインポート
from pandas_datareader.yahoo.daily import YahooDailyReader
from datetime import datetime

date_st = datetime(2010, 1, 1)
date_fn = datetime(2021, 4, 1)

df1 = YahooDailyReader('MSFT', date_st, date_fn).read()

symbols = ['AAPL', 'MSFT', 'GOOGL']

dfs = [YahooDailyReader(symbol, date_st, date_fn).read() for symbol in symbols]

dfs = []
for symbol in symbols:
  df = YahooDailyReader(symbol, date_st, date_fn).read()
  dfs.append(df)

df2 = pd.concat(dfs, axis=0, keys=symbols).unstack(0)

from prophet import Prophet

data = df1.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# インスタンス化
model = Prophet()

# 学習
model.fit(data)

# 学習データに基づいて未来を予測
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 予測結果の可視化
#model.plot(forecast)
#plt.show()

# トレンド性と周期性の抽出
#model.plot_components(forecast)
#plt.show()


import streamlit as st
# markdownで文章が書ける
st.markdown('# 株価予測')
fig1 = model.plot(forecast)
st.pyplot(fig1)


st.markdown('# トレンド性・周期性の抽出')
fig2 = model.plot_components(forecast)
st.pyplot(fig2)