import qstock as qs

# 获取沪深300指数高开低收、成交量、成交金额、换手率数据，index是日期
data = qs.get_data(code_list=['HS300'], start='20050408', end='20230324', freq='d')

# 删除名称列、排序并去除空值
data = data.drop(columns=['name']).sort_index().fillna(method='ffill').dropna()

# 插入日期列
data.insert(0, 'date', data.index)

# 将日期从datetime格式转换为str格式
data['date'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
data = data.reset_index(drop=True)

import talib

# 收盘价的斜率
data['slope'] = talib.LINEARREG_SLOPE(data['close'].values, timeperiod=5)
# 相对强弱指标
data['rsi'] = talib.RSI(data['close'].values, timeperiod = 14)
# 威廉指标值
data['wr'] = talib.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod=7)
# MACD中的DIF、DEA和MACD柱
data['dif'], data['dea'], data['macd'] = talib.MACD(data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
# 抛物线指标
data['sar'] = talib.SAR(data['high'].values, data['low'].values)
# 删除开盘价、最高价和最低价
data = data.drop(columns=['open','high','low']).fillna(method='ffill').dropna().reset_index(drop=True)


from tsfresh.utilities.dataframe_functions import roll_time_series

data_roll = roll_time_series(data, column_id='code', column_sort='date', max_timeshift=20, min_timeshift=5).drop(columns=['code'])

print(data_roll.shape)
print(data_roll.head(15))