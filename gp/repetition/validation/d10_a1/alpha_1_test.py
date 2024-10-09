from rqfactor import *
from rqfactor.indicators import MACD

# pre expression
'''
delta(delta(mul(ts_min(atr), delta(scale(mul(covariance(high, emaDiff), log(volume)))))))
'''

# in expression
'''
delta(delta(ts_min(atr) * delta(scale(covariance(high, emaDiff) * log(volume)))))
'''


def compute():

    d = 10
    a = 1

    # atr: 平均真实波幅
    tr = MAX(MAX(Factor('high') - Factor('low'), ABS(Factor('high') - REF(Factor('close'), 1))), ABS(Factor('low') - REF(Factor('close'), 1)))
    atr = MA(tr, 14)  # 14天的简单移动平均

    emaDiff = MACD().DIFF

    cov_high_emaDiff = COV(Factor('high'),emaDiff,d)

    sclaed = SCALE(cov_high_emaDiff * LOG(Factor('volume')), a)

    alpha_1 = DELTA(DELTA(TS_MIN(atr, d) * DELTA(sclaed, d), d), d)

    return alpha_1