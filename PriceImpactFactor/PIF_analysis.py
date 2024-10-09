from PriceImpactFactor import *
import pandas as pd
import datetime
from rqfactor import *
import rqdatac

rqdatac.init()
d1 = '20210101'
d2 = '20211101'
f = Factor('pe_ratio_ttm')

ids = rqdatac.index_components('000300.XSHG',d1)
df = execute_factor(f, ids, d1, d2)

engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
result = engine.analysis(df, returns='daily', ascending=True, periods=1, keep_preprocess_result=True)
result['rank_ic_analysis'].summary()
