import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

### Raw data files comes from
#https://www.ncei.noaa.gov/access/ocean-carbon-data-system/oceans/CFC_ATM_Hist2015.html
#https://cdiac.ess-dive.lbl.gov/ftp/oceans/CFC_ATM_Hist/CFC_ATM_Hist_2015/


### Read in the Busenberg et al. 2015 Data Compilation
### Only goes to 2015
bsg = pd.read_csv('CFC_atmospheric_histories_revised_2015_Table1.csv', header=0)
units = bsg.loc[0]
bsg.drop(0, inplace=True)
# Get rid of fractional Years
# Index by integer year
bsg.index = bsg.Year.map(lambda x: np.floor(x).astype(int))


### Read in NOAA HATS data 
### Data up to 2020
def read_HATS(fname):
    # Find where the actual data begins
    with open(fname, 'r') as ff:
        lines = ff.readlines()
        ind = None
        for i in range(len(lines)):
            try:
                np.array(lines[i].split()[:5]).astype(float)
                ind = i
                break
            except ValueError:
                pass
            
    # Create a dataframe
    df = pd.read_csv(fname, skiprows=ind-1, delim_whitespace=True)
    df.index = pd.to_datetime(df.iloc[:,0].astype(str) + df.iloc[:,1].astype(str), format='%Y%m')
    return df


# Import the data
cfc11_ = 'HATS_global_F11.txt'
cfc11_hats = read_HATS(cfc11_)

cfc12_ = 'HATS_global_F12.txt'
cfc12_hats = read_HATS(cfc12_)

cfc113_ = 'HATS_global_F113.txt'
cfc113_hats = read_HATS(cfc113_)

sf6_ = 'HATS_global_SF6.txt'
sf6_hats = read_HATS(sf6_)

# Merge dataframes into single
cfc_11_12 = pd.merge(cfc11_hats.iloc[:,2], cfc12_hats.iloc[:,2], how='outer', left_index=True, right_index=True)
cfc_hats = pd.merge(cfc_11_12, cfc113_hats.iloc[:,2], how='outer', left_index=True, right_index=True)
hats = pd.merge(cfc_hats, sf6_hats.iloc[:,2], how='outer', left_index=True, right_index=True)

# Yearly average to match Busenberg data
hats_yr = hats.groupby([hats.index.year]).mean()


### Merge Busenberg and HATS dataframes
df = pd.merge(hats_yr, bsg[['CFC11NH','CFC12NH','CFC113NH','SF6NH']], how='outer', left_index=True, right_index=True).astype(float)

### Now make one long timeseries
### Take average between two datasets
df['cfc11_ppt']  = np.nanmean(df[['CFC11NH', 'HATS_NH_F11']], axis=1)
df['cfc12_ppt']  = np.nanmean(df[['CFC12NH', 'HATS_NH_F12']], axis=1)
df['cfc113_ppt'] = np.nanmean(df[['CFC113NH', 'HATS_NH_F113']], axis=1)
df['sf6_ppt']    = np.nanmean(df[['SF6NH', 'GML_NH_SF6']], axis=1)
df.index.name = 'year'

### Save to csv
df[['cfc11_ppt','cfc12_ppt','cfc113_ppt','sf6_ppt']].to_csv('cfc_sf6_atmhist_1765-2021.csv')


### Plots
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(df.cfc11_ppt, marker='.', c='C0', label='CFC-11')
ax.plot(df.cfc12_ppt, marker='.', c='C1', label='CFC-12')
ax.plot(df.cfc113_ppt, marker='.', c='C2', label='CFC-113')
ax2 = ax.twinx()
ax2.plot(df.sf6_ppt, marker='.', c='C3', label=r'$\mathrm{SF_{6}}$')
ax.set_xlabel('Year')
ax.set_ylabel('CFC-11, CFC-12, CFC-113 (PPT)')
ax2.set_ylabel(r'$\mathrm{SF_{6} \ (PPT)}$')
ax.set_xlim(1920,2020)
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.margins(x=0.01)
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
plt.show()




