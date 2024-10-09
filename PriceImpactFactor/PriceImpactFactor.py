import numpy as np
import pandas as pd
import rqdatac
from rqfactor import *
from rqfactor.interface import UserDefinedLeafFactor

rqdatac.init()

# 自定义因子算子声明
def get_factor(order_book_ids, start_date, end_date):
    """参数说
        order_book_ids: 股票列表
        start_date: 开始日期
        end_date: 结束日期
        返回结果要求：pandas.DataFrame （行索引index为date序列, 列索引columns为order_book_id序列）
    """

    # 设置阈值乘数
    threshold_multiplier = 2

    # 获取逐笔成交数据
    tick_data = rqdatac.get_price(order_book_ids, start_date, end_date, frequency='tick', fields=['trading_date', 'last', 'volume'])

    # 计算当天的每笔交易量
    tick_data['trade_volume'] = tick_data['volume'].diff().fillna(tick_data['volume'])
    # 使用累计成交量的差值计算每笔交易的实际成交量。对于第一笔交易，使用原始值填充NaN。

    # 计算大额交易的阈值
    volume_mean = tick_data['trade_volume'].mean()  # 计算每笔交易量的均值
    volume_std = tick_data['trade_volume'].std()  # 计算每笔交易量的标准差
    threshold = volume_mean + threshold_multiplier * volume_std
    # 大额交易阈值定义为均值加上阈值乘数乘以标准差

    # 标记大额交易
    tick_data['large_trade'] = tick_data['trade_volume'] > threshold
    # 如果某笔交易量大于阈值，则标记为大额交易

    # 计算每一天的价格冲击因子值
    tick_data['date'] = tick_data['trading_date'].dt.date
    result = tick_data.groupby('date').apply(calculate_price_impact)

    # index 转换为 pd.DatetimeIndex
    result.index = pd.to_datetime(result.index)
    return result


# 计算价格冲击
def calculate_price_impact(data):
    impacts = []
    for i in range(len(data)):
        if data.iloc[i]['large_trade']:
            before_window = data.iloc[max(0, i - 5):i]  # 大额交易前5笔交易的窗口
            after_window = data.iloc[i + 1:min(len(data), i + 6)]  # 大额交易后5笔交易的窗口

            if not before_window.empty and not after_window.empty:
                P_before = before_window['last'].mean()  # 大额交易前窗口的平均价格
                P_after = after_window['last'].mean()  # 大额交易后窗口的平均价格
                impact = (P_after - P_before) / P_before  # 价格冲击的计算公式
                impacts.append(impact)  # 将计算的价格冲击值加入列表
    return np.median(impacts) if impacts else np.nan
    # 返回价格冲击的中位数，如果没有大额交易则返回NaN

PIF = UserDefinedLeafFactor('PriceImpactFactor', get_factor)
