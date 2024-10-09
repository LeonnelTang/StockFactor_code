import numpy as np
import pandas as pd
import qstock as qs
import matplotlib.pylab as plt
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

# 确保代码在主程序中运行
if __name__ == '__main__':
    # 获取苹果公司股票数据
    df = qs.get_data('AAPL', start='2018-01-01')
    df = df['high']

    # 创建 DataFrame，并将索引重置为普通列
    df_melted = pd.DataFrame({"high": df.copy()}).reset_index()

    # 创建 'date' 列，值与 'Date' 列相同
    df_melted["Date"] = df_melted["date"]

    # 添加 Symbols 列并赋值为 'AAPL'
    df_melted["Symbols"] = "AAPL"

    # 设置索引为 'Date'
    df_melted.set_index("Date", inplace=True)

    # 显示结果
    print(df_melted.head())

    # 滚动时间序列数据
    df_rolled = roll_time_series(df_melted, column_id="Symbols", column_sort="date", max_timeshift=20, min_timeshift=5)

    # 显示结果
    print(df_rolled.head())

    X = extract_features(df_rolled.drop("Symbols", axis=1),
                         column_id="id", column_sort="date", column_value="high",
                         impute_function=impute, show_warnings=False)

    X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
    X.index.name = "last_date"
    print(X.head())
