import numpy as np
import pandas as pd
import graphviz
from scipy.stats import rankdata
import pickle
import scipy.stats as stats
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 系统自带的函数群

"""
Available individual functions are:
‘add’ : addition, arity=2.
‘sub’ : subtraction, arity=2.
‘mul’ : multiplication, arity=2.
‘div’ : protected division where a denominator near-zero returns 1., arity=2.
‘sqrt’ : protected square root where the absolute value of the argument is used, arity=1.
‘log’ : protected log where the absolute value of the argument is used and a near-zero argument returns 0., arity=1.
‘abs’ : absolute value, arity=1.
‘neg’ : negative, arity=1.
‘inv’ : protected inverse where a near-zero argument returns 0., arity=1. 
‘max’ : maximum, arity=2.
‘min’ : minimum, arity=2.
‘sin’ : sine (radians), arity=1.
‘cos’ : cosine (radians), arity=1.
‘tan’ : tangent (radians), arity=1.
"""

# init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']
init_function = ['add', 'sub', 'mul', 'div', 'abs', 'sqrt', 'log', 'inv']


# 基础函数
def __rolling_rank(data):
    value = rankdata(data)[-1]
    return value


def __rolling_prod(data):
    return np.prod(data)


d = 10
a = 2


# 自定义函数，make_function 函数群
def _rank(data):
    value = pd.Series(data.flatten()).rank()
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def _delay(data):
    global d
    value = pd.Series(data.flatten()).shift(d)
    return np.nan_to_num(value)


def _pct(data):
    value = pd.Series(data.flatten()).pct_change()
    return np.nan_to_num(value)


def _correlation(data1, data2):
    global d
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            x1 = pd.Series(data1.flatten())
            x2 = pd.Series(data2.flatten())
            df = pd.concat([x1, x2], axis=1)
            temp = pd.Series(dtype='float64')
            for i in range(len(df)):
                if i <= d - 2:
                    temp[str(i)] = np.nan
                else:
                    df2 = df.iloc[i - d + 1:i + 1, :]
                    temp[str(i)] = df2.corr('spearman').iloc[1, 0]
            return np.nan_to_num(temp)
        except:
            return np.zeros(data1.shape[0])


def _covariance(data1, data2):
    global d
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            x1 = pd.Series(data1.flatten())
            x2 = pd.Series(data2.flatten())
            df = pd.concat([x1, x2], axis=1)
            temp = pd.Series(dtype='float64')
            for i in range(len(df)):
                if i <= d - 2:
                    temp[str(i)] = np.nan
                else:
                    df2 = df.iloc[i - d + 1:i + 1, :]
                    temp[str(i)] = df2.cov().iloc[1, 0]
            return np.nan_to_num(temp)
        except:
            return np.zeros(data1.shape[0])


def _scale(data):
    global a
    data = pd.Series(data.flatten())
    value = data.mul(a).div(np.abs(data).sum())
    return np.nan_to_num(value)


def _delta(data):
    global d
    value = pd.Series(data.flatten())
    delay = pd.Series(data.flatten()).shift(d)
    return np.nan_to_num(value - delay)


def _signedpower(data):
    global a
    value = pd.Series(data.flatten())
    value = np.sign(value) * abs(value) ** a
    return np.nan_to_num(value)


def _decay_linear(data):
    global d
    weight = np.arange(1, d + 1, 1)
    weight = weight / sum(weight)
    value = pd.Series(data.flatten())
    temp = pd.Series(dtype='float64')
    for i in range(len(value)):
        if i <= d - 2:
            temp[str(i)] = np.nan
        else:
            value_window = np.nan_to_num(value.iloc[i - d + 1:i + 1])
            temp[str(i)] = sum(value_window * weight)
    return np.nan_to_num(temp)


def _ts_min(data):
    global d
    value = np.array(pd.Series(data.flatten()).rolling(d).min().tolist())
    return np.nan_to_num(value)


def _ts_max(data):
    global d
    value = np.array(pd.Series(data.flatten()).rolling(d).max().tolist())
    return np.nan_to_num(value)


def _ts_argmin(data):
    global d
    value = pd.Series(data.flatten()).rolling(d).apply(np.argmin) + 1
    return np.nan_to_num(value)


def _ts_argmax(data):
    global d
    value = pd.Series(data.flatten()).rolling(d).apply(np.argmax) + 1
    return np.nan_to_num(value)


def _ts_rank(data):
    global d
    value = pd.Series(data.flatten()).rolling(d).apply(__rolling_rank)
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def _ts_sum(data):
    global d
    value = pd.Series(data.flatten()).rolling(d).sum()
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def _ts_product(data):
    global d
    value = pd.Series(data.flatten()).rolling(d).apply(__rolling_prod)
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def _ts_stddev(data):
    global d
    value = pd.Series(data.flatten()).rolling(d).std()
    value = np.array(value.tolist())
    return np.nan_to_num(value)


#自定义函数群
rank = make_function(function=_rank, name='rank', arity=1)
delay = make_function(function=_delay, name='delta', arity=1)
pct = make_function(function=_pct, name='pct', arity=1)
correlation = make_function(function=_correlation, name='correlation', arity=2)
covariance = make_function(function=_covariance, name='covariance', arity=2)
scale = make_function(function=_scale, name='scale', arity=1)
delta = make_function(function=_delta, name='delta', arity=1)
signedpower = make_function(function=_signedpower, name='signedpower', arity=1)
decay_linear = make_function(function=_decay_linear, name='decay_linear', arity=1)
ts_min = make_function(function=_ts_min, name='ts_min', arity=1)
ts_max = make_function(function=_ts_max, name='ts_max', arity=1)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=1)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=1)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=1)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=1)
ts_product = make_function(function=_ts_product, name='ts_product', arity=1)
ts_stddev = make_function(function=_ts_stddev, name='delta', arity=1)

my_function = [rank, delay, correlation, covariance, scale, delta, signedpower, decay_linear,
               ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, ts_sum, ts_stddev]

#导入数据
data = pd.read_csv('data.csv')
code = '399006.XSHE'
trainData = data[data.code == code].copy()
label = np.array(trainData.ret.iloc[1:])
features = np.array(trainData.drop(columns=['date', 'code', 'ret']).iloc[:-1, :])
fields = list(trainData.drop(columns=['date', 'code', 'ret']).columns)

#数据标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)
lable = scaler.fit_transform(label.reshape(-1, 1))

#遗传规划
generation = 3
function_set = init_function + my_function
population_size = 1000
random_state = 0
est_gp = SymbolicTransformer(feature_names=fields
                             , function_set=function_set
                             , generations=generation
                             , population_size=population_size
                             , tournament_size=20
                             , random_state=random_state
                             , verbose=1
                             , init_depth=(1, 4)
                             , p_crossover=0.4
                             , p_subtree_mutation=0.01
                             , p_hoist_mutation=0
                             , p_point_mutation=0.01
                             , p_point_replace=0.4)
est_gp.fit(features, label)

# 获取较优的表达式
best_programs = est_gp._best_programs
best_programs_dict = {}
for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                       'length': p.length_}

best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness', ascending=False)
print(best_programs_dict)

# 保存为 CSV 文件
best_programs_dict.to_csv('best_programs_dict_d10_a2.csv', index=True)

'''
def alpha_factor_graph(num):
    # 打印指定num的表达式图

    factor = best_programs[num - 1]
    print(factor)
    print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('images/alpha_factor_graph', format='png', cleanup=True)

    return graph


# 打印因子alpha的结构图
graph10 = alpha_factor_graph(10)
print(graph10)
'''