from rqfactor import *


def compute():
    wap = (Factor('high')+Factor('low')+Factor('close')) / 3
    lci = (wap - Factor('high'))/(Factor('high') - Factor('low')) * Factor('volume')
    normLci = (lci - MA(lci,30)) / STD(lci,30)
    relativePriceMomentum = (Factor('close') - REF(Factor('close'),20)) / REF(Factor('close'),20) * 100
    dlcf = -normLci * relativePriceMomentum

    # EMA_CN: 指数移动平均, 除以Factor('close')转换为无量纲因子
    smoothedDlcf = (EMA_CN(dlcf, 12))
    return smoothedDlcf
