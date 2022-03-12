import pandas as pd

# read data
dat = pd.read_excel("../INput/Problem3.5_updatedversion_data.xlsx",skiprows=1).iloc[:,:7]
dat.columns = ['yyyymm'] + list(dat.columns[1:])
