# -*- coding: utf-8 -*-
"""
Created on Wed May 25 07:40:15 2022

@author: n2ngo
"""

import pandas as pd

dat = pd.read_csv("../Input/data_all.csv")

# reverse sign of RepRisk score
dat['current_rri'] *= -1

dat_subset = dat[dat['jp_flag']==1]

# correlation
dat_corr = dat_subset[['current_rri','TRESGScore', 'MSCI_KLD_raw_ESG_Score']].dropna(axis=0)
dat_corr.columns = ['RepRisk','Refinitiv','MSCI KLD']
dat_corr.corr()

# OLS
# dat_ols = dat_subset[['gvkey','ret_f1M','rf_f1M','current_rri','TRESGScore', 'MSCI_KLD_raw_ESG_Score','log_size','book_to_market','ebit_over_totalasset','Beta','volatility','momentum','sic_desc','fic','yyyymm']]
dat_ols = dat_subset[['gvkey','ret_f1M','rf','current_rri','TRESGScore', 'MSCI_KLD_raw_ESG_Score','log_size','book_to_market','ebit_over_totalasset','Beta','volatility','momentum','sic_desc','fic','yyyymm']]

dat_ols =  dat_ols.dropna(axis=0)
dat_ols['yyyymm'] = dat_ols['yyyymm'].astype(str)
dat_ols.groupby(['yyyymm'])['gvkey'].count().plot(rot=45)

# 2SLS