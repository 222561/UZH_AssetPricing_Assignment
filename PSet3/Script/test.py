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
def MeanVarianceFrontier_set(mu,cov,color,label):
    def element_MeanVarianceFrontier(mu,cov):
        invcov = np.linalg.inv(cov)
        ones =  mu*0+1
        
        A = np.dot(np.dot(mu,invcov),mu)
        B = np.dot(np.dot(ones,invcov),mu)
        C = np.dot(np.dot(ones,invcov),ones)
        D = A*C-B**2
        
        return A,B,C,D
    
    def MeanVarianceFrontier(sigma,A,B,C,D):
        if sigma**2-1/C > 0:
            mu_p = B/C + np.sqrt(D/C*(sigma**2-1/C))
        else:
            mu_p = B/C
        return mu_p
    
    A,B,C,D = element_MeanVarianceFrontier(mu,cov)

    sigmas = list(np.arange(np.sqrt(1/C),15,step=0.05))
    mu_ps = [MeanVarianceFrontier(sigma,A,B,C,D) for sigma in sigmas]
    

    plt.scatter(sigmas,mu_ps,color=color,label = label)
        
    plt.legend()
    
    

def MeanVarianceFrontier_set_w_rf(mu,cov,color,label):
    invcov = np.linalg.inv(cov)
    E = np.dot(np.dot(mu,invcov),mu)
    
    sigmas = list(np.arange(0,15,step=0.05))
    mu_ps = [np.sqrt(E)*sigma for sigma in sigmas]

    plt.scatter(sigmas,mu_ps,color=color,label = label)
        
    plt.legend()
    
    # get tangent portfolio's SR
    SR = np.divide(mu_ps,sigmas)[-1]
    
    return SR

    
def run_for_i(syyyymm,eyyyymm):
    sample =  dat[(dat['yyyymm']>=syyyymm) & (dat['yyyymm']<=eyyyymm)].set_index('yyyymm')
    
    # mean excess returns (over riskfree rate)
    print("#-------------------#")
    print("Mean excess returns (over riskfree rate)")
    exrtn = sample.copy()
    for col in exrtn.columns:
        exrtn[col] = exrtn[col] - sample['Riskfree Rate']
    mean_exrtn = exrtn.mean(axis = 0)
    display(mean_exrtn.round(2))
    
    # covariance of excess return
    print("#-------------------#")
    print("Covariance of excess return")
    cov = pd.DataFrame(np.cov(exrtn.T))
    cov.index = mean_exrtn.index
    cov.columns = mean_exrtn.index
    display(cov)
    
    
    print("#-------------------#")
    print("Plot of Mean Variance Frontiers")
    mu = mean_exrtn[:-1]
    cov = cov.iloc[:-1,:-1]
    MeanVarianceFrontier_set(mu,cov,"green","MV fronter without riskless-asset")
    SR_tangent = MeanVarianceFrontier_set_w_rf(mu,cov,"blue","MV fronter with riskless-asset")
    
    colors = ['red','yellow','orange','purple','grey']
    for i in range(cov.shape[0]):
        if mu.index[i] == "Market ":
            plt.scatter(np.sqrt(cov.iloc[i,i]),mu[i],color=colors[i],label = mu.index[i],marker='^')
        else:
            plt.scatter(np.sqrt(cov.iloc[i,i]),mu[i],color=colors[i],label = mu.index[i],marker='*')
        plt.legend()
    plt.legend() 
    
     # Calculate the Sharpe ratios of the tangency portfolio and the market portfolio
    print("#-------------------#")
    print(" the Sharpe ratios of the tangency portfolio: " + str(round(SR_tangent,3)))
    SR_market = exrtn['Market '].mean()/exrtn['Market '].std()
    print(" the Sharpe ratios of the market portfolio: " + str(round(SR_market,3)))
    
    return SR_tangent,SR_market
    

syyyymm,eyyyymm = 192607, 202106
run_for_i(syyyymm,eyyyymm)

# # mean - variance efficent sets
# def exret_vol(port_weights):
#     port_exret = np.dot(mean_exrtn,port_weights)
#     port_vol = np.sqrt(np.dot(port_weights,np.dot(cov,port_weights)))
#     return port_exret, port_vol

# def port_SR(port_weights):
#     port_exret, port_vol = exret_vol(port_weights)
#     return port_exret/port_vol

# def objective(port_weights):
#     return -port_SR(port_weights)

# def optimized_weights(sample,objective,bnds,cons):
#     port_weights = (np.zeros(sample.shape[1]) + 1)/sample.shape[1]
#     res = minimize(objective, port_weights, method='SLSQP',
#                    bounds=bnds,constraints=cons)
    
#     port_weights = list(res.x)
#     return port_weights

# # without riskless asset
# # bnds = []
# bnds = [(0,1) for i in range(sample.shape[1])]
# cons = [{'type':'eq','fun': lambda port_weights: np.sum(port_weights) - 1},
#         {'type':'eq','fun': lambda port_weights: port_weights[-1] }]
# weights_opt1 = optimized_weights(sample,objective,bnds,cons)
# exret_vol(weights_opt1)
# port_SR(weights_opt1)

# # with riskless asset
# cons = [{'type':'eq','fun': lambda port_weights: np.sum(port_weights) - 1}]
# weights_opt2 = optimized_weights(sample,objective,bnds,cons)
# exret_vol(weights_opt2)
# port_SR(weights_opt2)

# df = pd.DataFrame([exret_vol([1,0,0,0,0,0])
#                    ,exret_vol([0,1,0,0,0,0])
#                    ,exret_vol([0,0,1,0,0,0])
#                    ,exret_vol([0,0,0,1,0,0])
#                    ,exret_vol([0,0,0,0,1,0])
#                    ,exret_vol([0,0,0,0,0,1])
#                    ,exret_vol(weights_opt1)
#                    ,exret_vol(weights_opt2)])
# df.index = list(sample.columns) + ['Without riskless asset','With riskless asset']
# # df.plot.scatter(x=1,y=0,xlim=(0,10),ylim=(-0.1,1.5))
# # plt.close()

# # plot two sets above + each factor portfolio (x-axis: std, y-axis: mean excess return
# for i,row in enumerate(df.index):
#     if i == 4:
#         plt.scatter(df.loc[row,1],df.loc[row,0],marker="s")
#         plt.text(df.loc[row,1]+0.2,df.loc[row,0]-0.04, row)
#     elif i <4:
#         plt.scatter(df.loc[row,1],df.loc[row,0])
#         plt.text(df.loc[row,1]-1,df.loc[row,0]-0.1, row)
#     else:
#         plt.scatter(df.loc[row,1],df.loc[row,0],marker="^")
#         plt.text(df.loc[row,1]-1.5,df.loc[row,0]+0.05, row)
# plt.show()
# plt.close()


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
plt.show()
plt.close()
    
# plot beta - alpha
for col in sample.columns[:4]:
    alpha,beta,exret = calc_CAPM(sample,col)
    plt.scatter(beta,alpha)
    plt.text(beta-0.025,alpha+0.01, col)
plt.show()
plt.close()

# iii)
#  Gibbons-Ross-Shanken test
# reference: https://github.com/SoniaistSonia/GRS-test_Python/blob/master/GRS_test_Python

"""
Function GRS_test(factor, resid, alpha) is to conduct GRS test according 
to Gibbons, Ross & Shanken(1989) to receive GRS-statistic and p-value.

H0: alpha1=alpha2=...=alphaN

Parameters:
  T = number of months
  N = number of portfolios
  L = number of factors

Inputs:
  factor: matrix of FF factors with shape (T, L)
  resid: matrix of residuals with shape (T, N)
  alpha: matrix of intercepts with shape (N, 1)

Outputs:
  f_grs: GRS-statistic
  p_grs: P-value

"""

import scipy.stats as st

def GRS_test(factor, resid, alpha):
    N = resid.shape[1]        
    T = resid.shape[0]       
    L = factor.shape[1]      

    if (T-N-L) < 0:
        print('can not conduct GRS test because T-N-L<0')
        return

    factor = np.asmatrix(factor)                   # factor matrix (T, L)
    resid = np.asmatrix(resid)                     # residual matrix (T, N)
    alpha = np.asmatrix(alpha).reshape(N, 1)       # intercept matrix (N, 1)

    mean_return_factor = (factor.mean(axis=0))

    # covariance matrix of residuals
    cov_resid = (resid.T * resid) / (T-L-1)
    # covariance matrix of factors
    cov_factor = ((factor - mean_return_factor).T * (factor - mean_return_factor)) / (T-1)

    mean_return_factor = mean_return_factor.reshape(L, 1)

    # GRS statistic
    f_grs = float((T/N) * ((T-N-L)/(T-L-1)) * ((alpha.T * np.linalg.inv(cov_resid) * alpha) / (1 + mean_return_factor.T * np.linalg.inv(cov_factor) * mean_return_factor)))

    # p-value
    p_grs = 1 - st.f.cdf(f_grs, N, (T-N-L))

    return f_grs, p_grs

 
# defining the variables
x = sample['Market '].tolist()
y = np.dot(sample,weights_opt2).tolist()
 
# adding the constant term
x = sm.add_constant(x)
 
# performing the regression
# and fitting the model
result = sm.OLS(y, x).fit()


GRS_test(pd.DataFrame(sample['Market ']),pd.DataFrame(result.resid),pd.DataFrame([result.params[0]]))


# =============================================================================
# # b)
# =============================================================================
# i)
# 1y MA of small-high minus small-low

def plot_12MA__smallHML(syyyymm,eyyyymm):
    sample =  dat[(dat['yyyymm']>=syyyymm-100) & (dat['yyyymm']<=eyyyymm)].set_index('yyyymm')
    sample.index = pd.to_datetime(sample.index, format='%Y%m')
    sample['small HML'] = (sample['Small-High']-sample['Small-Low'])
    sample['small HML'].rolling(12).mean().plot()

# main period
syyyymm,eyyyymm = 196401,202106
plot_12MA__smallHML(syyyymm,eyyyymm)
plt.show()
plt.close()
# 4 sub periods
syyyymm,eyyyymm = 196401,199312
plot_12MA__smallHML(syyyymm,eyyyymm)
syyyymm,eyyyymm = 199401,200912
plot_12MA__smallHML(syyyymm,eyyyymm)
syyyymm,eyyyymm = 201001,202001
plot_12MA__smallHML(syyyymm,eyyyymm)
syyyymm,eyyyymm = 202002,202106
plot_12MA__smallHML(syyyymm,eyyyymm)
plt.show()
plt.close()

# ii)
# calculate ret,std,SR on small-High 
def calc_performance_smallHML(syyyymm,eyyyymm):
    sample =  dat[(dat['yyyymm']>=syyyymm) & (dat['yyyymm']<=eyyyymm)].set_index('yyyymm')
    sample.index = pd.to_datetime(sample.index, format='%Y%m')
    sample['small HML'] = (sample['Small-High']-sample['Small-Low'])
    ret = sample['small HML'].mean()
    std = sample['small HML'].std()
    SR = ret/std
    return ret,std,SR

# main period
syyyymm,eyyyymm = 196401,202106
summary = pd.DataFrame(calc_performance_smallHML(syyyymm,eyyyymm))
summary.columns = [str(syyyymm)+"-"+str(eyyyymm)]
summary.index = ['Return','Std','SR']
# 4 sub periods
syyyymm,eyyyymm = 196401,199312
summary[str(syyyymm)+"-"+str(eyyyymm)] = calc_performance_smallHML(syyyymm,eyyyymm)
syyyymm,eyyyymm = 199401,200912
summary[str(syyyymm)+"-"+str(eyyyymm)] = calc_performance_smallHML(syyyymm,eyyyymm)
syyyymm,eyyyymm = 201001,202001
summary[str(syyyymm)+"-"+str(eyyyymm)] = calc_performance_smallHML(syyyymm,eyyyymm)
syyyymm,eyyyymm = 202002,202106
summary[str(syyyymm)+"-"+str(eyyyymm)] = calc_performance_smallHML(syyyymm,eyyyymm)
summary.T

# iii)
# calculate quartely return on small-High
def cumret_f(x):
    ret = 1
    for i in x:
        ret *= 1+i/100
    return (ret  - 1)*100

def acf_sHML_quartely_ret(syyyymm,eyyyymm):
    sample =  dat[(dat['yyyymm']>=syyyymm) & (dat['yyyymm']<=eyyyymm)].set_index('yyyymm')
    sample['Small HML'] = (sample['Small-High']-sample['Small-Low'])
    sample['Small-HML_Q'] = sample['Small HML'].rolling(3).apply(cumret_f)
    sample = sample.iloc[[True if str(yyyymm)[-2:] in ['03','06','09','12'] else False for yyyymm in sample.index],:]
    
    sample.index = pd.to_datetime(sample.index, format='%Y%m')
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(sample['Small-HML_Q'], lags=np.arange(13), title="ACF: " + str(syyyymm) + "-" + str(eyyyymm))

# calculate autocorrelation
# syyyymm,eyyyymm = 196401,202106
# acf_sHML_quartely_ret(syyyymm,eyyyymm)
syyyymm,eyyyymm = 196401,199312
acf_sHML_quartely_ret(syyyymm,eyyyymm)
syyyymm,eyyyymm = 199401,200912
acf_sHML_quartely_ret(syyyymm,eyyyymm)
syyyymm,eyyyymm = 201001,202106
acf_sHML_quartely_ret(syyyymm,eyyyymm)



# iv)
# What do your results suggest about the changing behavior of value returns in recent 
# years? Do they support any of the concerns described above?




