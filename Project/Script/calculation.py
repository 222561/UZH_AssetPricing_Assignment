# -*- coding: utf-8 -*-
"""
Created on Wed May 25 07:40:15 2022

@author: n2ngo
"""

import pandas as pd
import os 
import numpy as np
import statsmodels.api as sm


os.chdir("C:/Users/n2ngo/iCloudDrive/11.UZH/06_2022FS/1_Asset Pricing/UZH_AssetPricing_Assignment/Project/Script")
dat = pd.read_csv("../Input/data_all.csv")

# reverse sign of RepRisk score
dat['current_rri'] *= -1

# scale MSCI KLD in (0,100)
dat['MSCI_KLD_raw_ESG_Score'] = (dat['MSCI_KLD_raw_ESG_Score'] + 1)/2*100 

# calculate excess return
dat['exret_f1M'] = (dat['ret_f1M'] - dat['rf_f1M'])*100

def main_ols(country,dat):
    dat_subset = dat[dat[country+'_flag']==1].copy()
    
    # correlation
    dat_corr = dat_subset[['current_rri','TRESGScore', 'MSCI_KLD_raw_ESG_Score']].dropna(axis=0)
    dat_corr.columns = ['RepRisk','Refinitiv','MSCI KLD']
    print(dat_corr.corr())
    
    # OLS
    # make additional data
    sectorFE = pd.get_dummies(dat_subset['sic_desc'],drop_first=False) #i don't drop first since some rows have np.nan
    countryFE = pd.get_dummies(dat_subset['fic'],drop_first=False) #i don't drop first since some rows have np.nan
    monthFE = pd.get_dummies(dat_subset['yyyymm'],drop_first=True)
    
    col_esg = ['current_rri','TRESGScore', 'MSCI_KLD_raw_ESG_Score']
    esg_dict = dict(zip(col_esg,['RepRisk','Refinitiv','MSCI KLD']))
    
    col_stock_level = ['log_size','book_to_market','ebit_over_totalasset','Beta','volatility','momentum']
    
    
    dat_ols = dat_subset[['gvkey','yyyymm','exret_f1M']+col_esg+col_stock_level]
    if country =='eu':
        dat_ols = pd.concat([dat_ols, sectorFE,countryFE,monthFE],axis=1).dropna(axis=0)
        col_FE = list(sectorFE.columns) + list(countryFE.columns) + list(monthFE.columns)
    else:
        dat_ols = pd.concat([dat_ols, sectorFE,monthFE],axis=1).dropna(axis=0)
        col_FE = list(sectorFE.columns) + list(monthFE.columns)
    
    ret = pd.DataFrame()
    
# =============================================================================
#     # OLS
# =============================================================================
    for i in range(len(col_esg)):
        X = dat_ols[[col_esg[i]]+col_stock_level+col_FE].copy()
        X = sm.add_constant(X)
        y = dat_ols['exret_f1M'].copy()            
        
        result = sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'time':dat_ols['yyyymm'],'group':dat_ols['gvkey'],'maxlags':1})
        
        tmp = pd.DataFrame(['OLS',country.upper(),esg_dict[col_esg[i]],list(result.params)[1], list(result.bse)[1] ,list(result.tvalues)[1]]).T
        tmp.columns=['Model','Region','ESG Score','Coeffs','StdErr','t-stat']
        
        ret = pd.concat([ret,tmp],axis=0)
        
# =============================================================================
#     #2SLS
# =============================================================================
    for i in range(len(col_esg)):
        #1SLS
        y = dat_ols[col_esg[i]].copy()
        X = dat_ols[[col for col in col_esg if col != col_esg[i]]+col_stock_level+col_FE].copy()
        X = sm.add_constant(X)
        
        result = sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'time':dat_ols['yyyymm'],'group':dat_ols['gvkey'],'maxlags':1})
        
        X = pd.concat([result.predict(X).to_frame().rename(columns={0:col_esg[i]}),dat_ols[col_stock_level+col_FE]],axis=1).copy()
        
        X = sm.add_constant(X)
        y = dat_ols['exret_f1M'].copy()            
        
        result = sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'time':dat_ols['yyyymm'],'group':dat_ols['gvkey'],'maxlags':1})
        
        tmp = pd.DataFrame(['2SLS',country.upper(),esg_dict[col_esg[i]],list(result.params)[1], list(result.bse)[1] ,list(result.tvalues)[1]]).T
        tmp.columns=['Model','Region','ESG Score','Coeffs','StdErr','t-stat']
        
        ret = pd.concat([ret,tmp],axis=0)
    
    return ret.reset_index(drop=True)

res_ols = pd.DataFrame()
res_ols = pd.concat([res_ols,main_ols('us',dat)],axis=0)
res_ols = pd.concat([res_ols,main_ols('eu',dat)],axis=0)
res_ols = pd.concat([res_ols,main_ols('jp',dat)],axis=0)
res_ols
