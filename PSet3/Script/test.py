import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# read data
dat = pd.read_excel("../INput/Problem3.5_updatedversion_data.xlsx",skiprows=1).iloc[:,:7]
dat.columns = ['yyyymm'] + list(dat.columns[1:])

#
# =============================================================================
# # a)
# =============================================================================
# i) 
syyyymm,eyyyymm = 192607, 196312
sample =  dat[(dat['yyyymm']>=syyyymm) & (dat['yyyymm']<=eyyyymm)].drop('yyyymm',axis=1)

# mean excess returns (over riskfree rate)
mean_exrtn = sample.mean(axis = 0) - sample.mean(axis=0).iloc[-1] 
mean_exrtn.round(2)

# covariance of excess return
cov = pd.DataFrame(np.cov(sample.T))
cov.index = mean_exrtn.index
cov.columns = mean_exrtn.index
cov

# mean - variance efficent sets
def exret_vol(port_weights):
    port_exret = np.dot(mean_exrtn,port_weights)
    port_vol = np.sqrt(np.dot(port_weights,np.dot(cov,port_weights)))
    return port_exret, port_vol

def port_SR(port_weights):
    port_exret, port_vol = exret_vol(port_weights)
    return port_exret/port_vol

def objective(port_weights):
    return -port_SR(port_weights)

def optimized_weights(sample,objective,bnds,cons):
    port_weights = (np.zeros(sample.shape[1]) + 1)/sample.shape[1]
    res = minimize(objective, port_weights, method='SLSQP',
                   bounds=bnds,constraints=cons)
    
    port_weights = list(res.x)
    return port_weights

# without riskless asset
# bnds = []
bnds = [(0,1) for i in range(sample.shape[1])]
cons = [{'type':'eq','fun': lambda port_weights: np.sum(port_weights) - 1},
        {'type':'eq','fun': lambda port_weights: port_weights[-1] }]
weights_opt1 = optimized_weights(sample,objective,bnds,cons)
exret_vol(weights_opt1)
port_SR(weights_opt1)

# with riskless asset
cons = [{'type':'eq','fun': lambda port_weights: np.sum(port_weights) - 1}]
weights_opt2 = optimized_weights(sample,objective,bnds,cons)
exret_vol(weights_opt2)
port_SR(weights_opt2)

df = pd.DataFrame([exret_vol([1,0,0,0,0,0])
                   ,exret_vol([0,1,0,0,0,0])
                   ,exret_vol([0,0,1,0,0,0])
                   ,exret_vol([0,0,0,1,0,0])
                   ,exret_vol([0,0,0,0,1,0])
                   ,exret_vol([0,0,0,0,0,1])
                   ,exret_vol(weights_opt1)
                   ,exret_vol(weights_opt2)])
df.plot.scatter(x=1,y=0,xlim=(0,10),ylim=(-0.1,1.5))
# plt.close()

# plot two sets above + each factor portfolio (x-axis: std, y-axis: mean excess return

# ii)
# calculate beta
import statsmodels.api as sm
def calc_CAPM(sample,col):
    y = sample[col] - sample['Riskfree Rate']
    x = sample['Market '] - sample['Riskfree Rate']
    x = sm.add_constant(x)
    result = sm.OLS(y, x).fit()
    alpha = result.params.iloc[0]
    beta = result.params.iloc[1]
    exret_beta = (beta*x.drop('const',axis=1)).mean()
    return alpha, beta, exret_beta

# plot beta - expected return graph
for col in sample.columns[:4]:
    alpha,beta,exret = calc_CAPM(sample,col)
    plt.scatter(beta,exret)
    plt.text(beta-0.025,exret+0.03, col)
    
# plot beta - alpha
for col in sample.columns[:4]:
    alpha,beta,exret = calc_CAPM(sample,col)
    plt.scatter(beta,alpha)
    plt.text(beta-0.025,alpha+0.01, col)


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




