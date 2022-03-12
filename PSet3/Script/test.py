import pandas as pd

# read data
dat = pd.read_excel("../INput/Problem3.5_updatedversion_data.xlsx",skiprows=1).iloc[:,:7]
dat.columns = ['yyyymm'] + list(dat.columns[1:])

# make sample dataset
syyyymm,eyyyymm = 192607, 196312
sample =  dat[(dat['yyyymm']>=syyyymm) & (dat['yyyymm']<=eyyyymm)]

# =============================================================================
# # a)
# =============================================================================
# i) 
# mean excess returns

# covariance of excess return

# mean - variance efficent sets
# without riskless asset


# with riskless asset


# plot two sets above + each factor portfolio (x-axis: std, y-axis: mean excess return

# ii)
# calculate beta

# plot beta - expected return graph


# iii)
#  Gibbons-Ross-Shanken test


# =============================================================================
# # b)
# =============================================================================
# i)
# 1y MA of small-high minus small-low

# main period
# syyyymm,eyyyymm = 196401,202106
# 4 sub periods
# syyyymm,eyyyymm = 196401,199312
# syyyymm,eyyyymm = 199401,200912
# syyyymm,eyyyymm = 201001,202001
# syyyymm,eyyyymm = 202002,202106

# ii)
# calculate ret,std,SR on small-High 
# 4 sub periods


# iii)
# calculate quartely retu on small-High
# calculate autocorrelation
# syyyymm,eyyyymm = 196401,199312
# syyyymm,eyyyymm = 199401,200912
# syyyymm,eyyyymm = 201001,202106



# iv)
# What do your results suggest about the changing behavior of value returns in recent 
# years? Do they support any of the concerns described above?




