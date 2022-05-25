# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:52:03 2022

@author: n2ngo
"""
def make_id_list(df,col):
    string = ''
    for i,identifier in enumerate(df.dropna(subset=[col],axis=0)[col].unique()):
        if i == 0:
            string += "'" + str(identifier) + "'"
        else:
            string += ",'" + str(identifier) + "'"
    return string
# make_id_list(sp1200_hist,'gvkey')


def move_column(df,col,position):
    # take out a column
    column_to_move = df.pop(col)
    # insert column with insert(location, column_name, column_value)    
    df.insert(position, col, column_to_move)
    return df

def intersection(lst1, lst2):
    lst1_e = [value for value in lst1 if value not in lst2]
    lst12_i = [value for value in lst1 if value in lst2]
    lst2_e = [value for value in lst2 if value not in lst1]
    # return lst1_e, lst12_i, lst2_e
    return len(lst1_e), len(lst2_e), len(lst12_i)

import wrds
import pandas as pd
from matplotlib_venn import venn2
from matplotlib import pyplot
import numpy as np
from tqdm import tqdm

conn = wrds.Connection(wrds_username='tiwata')
# conn.list_libraries()
conn.list_tables('crsp_a_ccm')
# list_index = conn.get_table(library='comp_global', table='g_idx_index')
# list_index_us = conn.get_table(library='comp', table='idx_index')
# del list_index,list_index_us

# SPTSX '118341'
# SPX '000003'

# =============================================================================
# # get historical index data: sp global 1200 ex Canada
# =============================================================================
sp1200_hist = conn.raw_sql("select * from comp_global.g_idxcst_his where gvkeyx = '150918'")
speuro_hist = conn.raw_sql("select * from comp_global.g_idxcst_his where gvkeyx = '150927'")
spjpn_hist = conn.raw_sql("select * from comp_global.g_idxcst_his where gvkeyx = '151015'")


# spjpn_hist = conn.raw_sql("select * from comp_global.g_idxcst_his where gvkeyx = '151015'")

# sp500_hist = conn.raw_sql("select * from crsp.dsp500list")
# gv_permno = conn.raw_sql("select gvkey,lpermno as permno from crsp.ccm_lookup").dropna(axis=0).drop_duplicates()
# sp500_hist = pd.merge(gv_permno,sp500_hist,on='permno', how='right')


sp1200_hist['eu_flag'] = [1 if gvkey in list(speuro_hist.gvkey.unique()) else 0 for gvkey in sp1200_hist.gvkey]
sp1200_hist['jp_flag'] = [1 if gvkey in list(spjpn_hist.gvkey.unique()) else 0 for gvkey in sp1200_hist.gvkey]
# sp1200_hist['us_flg'] = [1 if gvkey in list(sp500_hist.gvkey.unique()) else 0 for gvkey in sp1200_hist.gvkey]

del speuro_hist


# get corresponding othe identifiers
ids_comp_g = conn.raw_sql("select * from comp_global.g_security where gvkey in (" + make_id_list(sp1200_hist,'gvkey') + ")")
# ids_crspcomp = conn.raw_sql("select gvkey,cusip,tic from crsp.ccm_lookup where gvkey in (" + make_id_list(sp1200_hist,'gvkey') + ")").drop_duplicates()

ids = pd.merge(sp1200_hist,
               ids_comp_g,
               on = 'gvkey',
               how = 'left')

del sp1200_hist
del ids_comp_g


sp500_hist = conn.raw_sql("select * from crsp.msp500list")
sp500_hist.columns = ['permno','from','thru']
gvkey_permno = conn.raw_sql("select gvkey, lpermno as permno from crsp.ccmxpf_linktable").dropna(axis=0).drop_duplicates()
sp500_hist = pd.merge(gvkey_permno,sp500_hist,on='permno', how='right')

ids_crsp = conn.raw_sql("select permno,cusip from crsp.dsfhdr where permno in  (" + make_id_list(sp500_hist,'permno') + ")")

sp500_hist = sp500_hist.merge(ids_crsp , on = 'permno',how='left')

gvkey_isin = conn.raw_sql("select gvkey,isin from crsp_a_ccm.sechead where gvkey in  (" + make_id_list(sp500_hist,'gvkey') + ")").dropna(axis=0).drop_duplicates()

sp500_hist = sp500_hist.merge(gvkey_isin , on = 'gvkey',how='left')

sp500_hist = sp500_hist.drop('permno',axis=1).dropna(subset=['gvkey'])
sp500_hist['us_flag'] = 1
sp500_hist['gvkeyx'] = 'sp500'

ids =pd.concat([ids,sp500_hist],axis=0)
del sp500_hist,ids_crsp

ids.to_csv("ids_all.csv",index=False)
# =============================================================================
# # get financial data
# =============================================================================
# na (compustat)
fin_na = conn.raw_sql("select gvkey,fyear,fyr,at,ebit,teq,fic,sich from comp.funda where gvkey in (" + make_id_list(ids,'gvkey') + ")")
fin_na['source'] = 'comp'

# non-na (compustat global)
fin_nonna = conn.raw_sql("select gvkey,fyear,fyr,at,ebit,teq,fic,sich from comp_global.g_funda where gvkey in (" + make_id_list(ids,'gvkey') + ")")
fin_nonna['source'] = 'comp_global'

test=conn.raw_sql("select * from comp_global.g_funda limit 5")

# venn diagram
venn2(subsets=(intersection(list(fin_na.gvkey.unique()), list(fin_nonna.gvkey.unique()))))
pyplot.title("financial data")
pyplot.show()

fin = pd.DataFrame()
for col in ['at','ebit','teq','fic']:
    tmp = pd.concat([fin_na,fin_nonna],axis=0).groupby(['gvkey','fyear','fyr'])[[col,'source']].first().rename(columns={'source':'source_'+col})
    fin = pd.concat([fin,tmp],axis=1)

del fin_na,fin_nonna,tmp,col

fin = fin.reset_index()
fin['yyyymm'] = fin['fyear'] * 100 + fin['fyr']
fin = move_column(fin,'yyyymm',1)

# =============================================================================
# # get market data
# =============================================================================
# na (compustat) (primiss:prime issue flag -> prioritize P (highest liquidity))
mkt_na    = conn.raw_sql("select gvkey, datadate, prccm as prcc, cshom as csho, ajexm as ajex, trfm as trf from comp.secm where primiss ='P' and gvkey in (" + make_id_list(ids,'gvkey') + ") ")
mkt_na['source'] = 'comp'
mkt_na['year'] = [Year.year for Year in mkt_na.datadate]
mkt_na['month'] = [Year.month for Year in mkt_na.datadate]

# non-na (compustat global) (prixxx:prime issue flag -> prioritize P (highest liquidity))
mkt_nonna = conn.raw_sql("select gvkey, datadate,prccd as prcc, cshoc as csho, ajexdi as ajex, trfd as trf from comp_global.g_secd where monthend = 1 and gvkey in (" + make_id_list(ids,'gvkey') + ") order by  tpci asc")
mkt_nonna['source'] = 'comp_global'
mkt_nonna['year'] = [Year.year for Year in mkt_nonna.datadate]
mkt_nonna['month'] = [Year.month for Year in mkt_nonna.datadate]

# venn diagram
venn2(subsets=(intersection(list(mkt_na.gvkey.unique()), list(mkt_nonna.gvkey.unique()))))
pyplot.title("market data")
pyplot.show()


mkt = pd.DataFrame()
for col in ['prcc','csho','ajex','trf']:
    tmp = pd.concat([mkt_na,mkt_nonna],axis=0).groupby(['gvkey','year','month'])[[col,'source']].first().rename(columns={'source':'source_'+col})
    mkt = pd.concat([mkt,tmp],axis=1)
del mkt_na,mkt_nonna,tmp,col

mkt['prc_for_return_calc'] = [ prcc / ajex * trf for prcc, ajex,trf in zip(mkt['prcc'],mkt['ajex'],mkt['trf'])]
mkt['adjprc'] = [ prcc / ajex for prcc, ajex in zip(mkt['prcc'],mkt['ajex'])]
mkt['mkt_cap']  = [ prcc *csho for prcc, csho in zip(mkt['prcc'],mkt['csho'])]
mkt = mkt.reset_index()
mkt['yyyymm'] = mkt['year']*100+mkt['month']
mkt = move_column(mkt,'yyyymm',1)

# =============================================================================
# # get reprisk esg data
# =============================================================================
repriskids = conn.raw_sql("select * from reprisk.v2_wrds_company_lookup where isin in (" + make_id_list(ids,'isin') + ")" )
repriskrri = conn.raw_sql("select * from reprisk.pm_rri_data where reprisk_id in (" + make_id_list(repriskids,'reprisk_id') + ")" )

repriskrri = pd.merge(repriskids,repriskrri,on='reprisk_id',how='left')
repriskrri['yyyymm'] = [date.year * 100+date.month if not isinstance(date, float) else np.nan for date in repriskrri.date]
del repriskids
repriskrri = move_column(repriskrri,'isin',0)
repriskrri = move_column(repriskrri,'yyyymm',1)
repriskrri = pd.merge(ids[['gvkey','isin']],repriskrri,on='isin',how='right')
repriskrri.pop('isin')
repriskrri= repriskrri.groupby(['gvkey','yyyymm'])[['current_rri']].max()
repriskrri = repriskrri.reset_index()


# =============================================================================
# # get refinitiv esg score : data comes from refinitiv eikon at UZH
# =============================================================================
def get_refinitivesg(key):
    refinitivesg = pd.read_csv("../data/refinitiv_esg_"+key+".csv")
    refinitivesg.columns.name = key
    refinitivesg = refinitivesg.set_index('date').stack(key).to_frame().rename(columns={0:'TRESGScore'}).replace("Unable to collect data for the field 'TR.TRESGScore' and some specific identifier(s).",np.nan).dropna(axis=0).reset_index()
    refinitivesg['yyyymm'] = [int(date[-4:]) * 100 + int(date[-7:-5]) *1 for date in refinitivesg.date]
    refinitivesg = pd.merge(ids[['gvkey',key]].dropna(axis=0).drop_duplicates(),refinitivesg,on=key,how='right')
    refinitivesg = refinitivesg[['gvkey','yyyymm','TRESGScore']].drop_duplicates()
    return refinitivesg

tresgscore_isin = get_refinitivesg('isin')
tresgscore_sedol = get_refinitivesg('sedol')

tresgscore = pd.concat([tresgscore_isin,tresgscore_sedol],axis=0)
tresgscore = tresgscore.groupby(['gvkey','yyyymm']).first()
tresgscore = tresgscore.reset_index()

del tresgscore_isin,tresgscore_sedol

# =============================================================================
# # make msci klg score (available only up to 2018)
# =============================================================================
# b= conn.raw_sql("select * from kld.kldnames")
# a.to_csv("check_kld.csv",index=False)

a=conn.raw_sql("select * from kld.history limit 1")

ecol_str = [col for col in a.columns  if (("env_str" in col) and ("_num" not in col))]
gcol_str = [col for col in a.columns  if (("cgov_str" in col) and ("_num" not in col))]
scol_str = [col for col in a.columns  if (("com_str" in col or "div_str" in col or "hum_str" in col or "pro_str" in col) and ("_num" not in col))]

ecol_con = [col for col in a.columns  if (("env_con" in col) and ("_num" not in col))]
gcol_con = [col for col in a.columns  if (("cgov_con" in col) and ("_num" not in col))]
scol_con = [col for col in a.columns  if (("com_con" in col or "div_con" in col or "hum_con" in col or "pro_con" in col) and ("_num" not in col))]

del a

msci_kld_all = conn.raw_sql("select * from kld.history")

# by cusip
list_cusip_gvkey = conn.raw_sql("select gvkey, cusip from ciq.wrds_cusip ws inner join ciq.wrds_gvkey wg on ws.companyid = wg.companyid").drop_duplicates()
list_cusip_gvkey['cusip'] = [cusip[:8] for cusip in list_cusip_gvkey['cusip']]
msci_kld_all_cusip = pd.merge(list_cusip_gvkey,msci_kld_all,on='cusip',how='right').dropna(subset=['gvkey'],axis=0)

# by ticker
list_ticker_gvkey = conn.raw_sql("select gvkey, ticker from ciq.wrds_ticker ws inner join ciq.wrds_gvkey wg on ws.companyid = wg.companyid").drop_duplicates()
msci_kld_all_ticker = pd.merge(list_ticker_gvkey,msci_kld_all,on='ticker',how='right').dropna(subset=['gvkey'],axis=0)

msci_kld_all_gvkey = pd.concat([msci_kld_all_cusip,msci_kld_all_ticker],axis=0).groupby(['gvkey','year']).first().reset_index()

del msci_kld_all_ticker,list_ticker_gvkey,msci_kld_all_cusip,list_cusip_gvkey
del msci_kld_all

msci_kld = msci_kld_all_gvkey[msci_kld_all_gvkey.gvkey.isin(list(ids.gvkey.unique()))]

def zero_if_denominator_zero(a,b):
    if b == 0:
        return 0
    elif a == 0:
        return 0
    else:
        return a/b

def count_msci_kld(series):
    """
    MSCI KLD based ESG score used in Lioui et al. 2021
    Parameters
    ----------
    series : TYPE
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    esc = series[ecol_str].count()
    ess = series[ecol_str].sum()
    ecc = series[ecol_con].count()
    ecs = series[ecol_con].sum()
    
    es = zero_if_denominator_zero(ess,esc)
    ec = zero_if_denominator_zero(ecs,ecc)
    
    gsc = series[gcol_str].count()
    gss = series[gcol_str].sum()
    gcc = series[gcol_con].count()
    gcs = series[gcol_con].sum()
    
    gs = zero_if_denominator_zero(gss,gsc)
    gc = zero_if_denominator_zero(gcs,gcc)

    ssc = series[scol_str].count()
    sss = series[scol_str].sum()
    scc = series[scol_con].count()
    scs = series[scol_con].sum()

    ss = zero_if_denominator_zero(sss,ssc)
    sc = zero_if_denominator_zero(scs,scc)

    res = pd.Series([esc,ess,ecc,ecs,gsc,gss,gcc,gcs,ssc,sss,scc,scs,
                     es,ec,gs,gc,ss,sc,
                     es-ec,gs-gc,ss-sc,
                     (es-ec+gs-gc+ss-sc)/3])
    res.index = ['e_str_count','e_str_sum','e_con_count','e_con_sum',
                 'g_str_count','g_str_sum','g_con_count','g_con_sum',
                 's_str_count','s_str_sum','s_con_count','s_con_sum',
                 'e_str_score','e_con_score',
                 'g_str_score','g_con_score',
                 's_str_score','s_con_score',
                 'e_score',
                 'g_score',
                 's_score',
                 'raw_esg_score'
                 ]
    return res



# 
msci_kld['MSCI_KLD_raw_ESG_Score'] = np.nan
for i in tqdm(msci_kld.index):
    msci_kld.loc[i,'MSCI_KLD_raw_ESG_Score'] = count_msci_kld(msci_kld.loc[i])[-1]
msci_kld['yyyymm'] = msci_kld.year * 100 + 1

# msci_kld[['gvkey','yyyymm']].drop_duplicates()


# =============================================================================
# # unfold historical index at monthly granularity
# =============================================================================
yyyymm=mkt.yyyymm.unique()
yyyymm.sort()
ids['from_yyyymm'] = [date.year *100 + date.month for date in ids['from']]
ids['to_yyyymm'] = [999912 if date ==None  else date.year *100 + date.month for date in ids['thru']]

index_hist = ids.assign(key=1).merge(pd.DataFrame(list(yyyymm)).rename(columns={0:'yyyymm'}).assign(key=1) , on ='key',how='left')
index_hist['index_flag'] = [1 if (yyyymm>=from_yyyymm) & (yyyymm<=to_yyyymm) else 0 for yyyymm,from_yyyymm,to_yyyymm in zip(index_hist['yyyymm'],index_hist['from_yyyymm'],index_hist['to_yyyymm'])]
index_hist = index_hist[['gvkey','yyyymm','index_flag','us_flag','eu_flag','jp_flag']].groupby(['gvkey','yyyymm']).max().reset_index()
for flag in ['us_flag','eu_flag','jp_flag']:
    index_hist[flag] = index_hist[flag] * index_hist['index_flag']
    index_hist[flag] = index_hist[flag].fillna(0)

# =============================================================================
# # merge all
# =============================================================================
dat_all = pd.merge(index_hist,fin, on = ['gvkey','yyyymm'],how='left')
dat_all = pd.merge(dat_all,mkt, on = ['gvkey','yyyymm'],how='left')
dat_all = pd.merge(dat_all,repriskrri[['gvkey','yyyymm','current_rri']], on = ['gvkey','yyyymm'],how='left')
dat_all = pd.merge(dat_all,tresgscore[['gvkey','yyyymm','TRESGScore']], on = ['gvkey','yyyymm'],how='left')
dat_all = pd.merge(dat_all,msci_kld[['gvkey','yyyymm','MSCI_KLD_raw_ESG_Score']], on = ['gvkey','yyyymm'],how='left')

# fill in blank between financial period (forward fullfilling is up to 15 months)
dat_all = dat_all.sort_values(['gvkey','yyyymm']).reset_index(drop=True)
dat_all = pd.concat([dat_all[['gvkey']],dat_all.groupby(['gvkey']).ffill(limit = 15)],axis=1)

# =============================================================================
# # calculate financial ratio
# =============================================================================

dat_all['log_size'] = [np.log(mktcap) if mktcap > 0 else np.nan for mktcap in dat_all['mkt_cap']]
dat_all['book_to_market'] = [teq*10**6/mktcap if (mktcap > 0 and teq > 0) else np.nan for teq,mktcap in zip(dat_all['teq'],dat_all['mkt_cap'])]
dat_all['ebit_over_totalasset'] = [ebit/at if (ebit > 0 and at > 0) else np.nan for ebit,at in zip(dat_all['ebit'],dat_all['at'])]

# =============================================================================
# # calculate ret(total return) momentum and volatility
# =============================================================================
dat_all['adjprc_f'] = dat_all.groupby(['gvkey'])[['adjprc']].shift(-1)
dat_all['ret_f1M'] = dat_all['adjprc_f']/dat_all['adjprc'] - 1
dat_all['momentum'] = dat_all.groupby(['gvkey'])[['adjprc']].pct_change(12)
dat_all['volatility'] = dat_all.groupby(['gvkey'])[['adjprc']].apply(lambda x : x.rolling(12).std())


# =============================================================================
# # cut data in period
# =============================================================================
dat_all['gvkey'] = dat_all['gvkey'].astype(str)
dat_all_cut = dat_all.query("yyyymm >= 201401")
dat_all_cut = dat_all_cut.query("yyyymm <= 202112")

# =============================================================================
# # save
# =============================================================================

dat_all_cut.to_csv("../Input/data_all.csv",index=False)
# ids.to_csv("../Input/historical_index_spglobal1200.csv",index=False)


dat_all_cut.groupby('yyyymm')[['index_flag','us_flag','eu_flag','jp_flag']].sum().plot(rot=45)

category_flag = ['index_flag','us_flag','eu_flag','jp_flag']
score_col = ['current_rri','TRESGScore','MSCI_KLD_raw_ESG_Score']
for category in category_flag:
    dat_all_cut[dat_all_cut[category]==1].groupby('yyyymm')[['gvkey'] + score_col].count().plot(rot=45,title=category)

dat_all_cut.groupby('yyyymm')[['index_flag','us_flag','eu_flag','jp_flag']].sum().plot(rot=45)

# index_hist.groupby('yyyymm')[['index_flag','us_flag','eu_flag','jp_flag']].sum().plot(rot=45)
# pd.read_csv("../Input/data_all.csv", dtype=str).head()
# pd.read_csv("../Input/historical_index_spglobal1200.csv", dtype=str).head()
