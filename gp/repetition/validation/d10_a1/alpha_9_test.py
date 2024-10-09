from rqfactor import *
from rqfactor.indicators import RSI

'''
sub(mul(money, atr), signedpower(rsrs))
'''

def compute():

    d = 10
    a = 1

    # atr: 平均真实波幅
    tr = MAX(MAX(Factor('high') - Factor('low'), ABS(Factor('high') - REF(Factor('close'), 1))),
             ABS(Factor('low') - REF(Factor('close'), 1)))
    atr = MA(tr, 14)  # 14天的简单移动平均

    # 总成交额
    money = Factor('total_turnover')

    # 相对强弱指标
    rsrs = RSI(6, 12, 24).RSI2

    alpha_9 = money * atr - SIGNEDPOWER(rsrs, a)

    return alpha_9