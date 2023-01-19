import numpy as np
import pandas as pd
import datetime
import cfc_tools
import sf6_tools
import stable_isotope_tools
import get_atm_conc

import datetime
import matplotlib.pyplot as plt
from scipy import stats


from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as mtick

plt.rcParams['font.size']=14
plt.rcParams['font.family']='serif'


from tu_2_molkg import *
tu2m = tu_2_molkg()
m2tu = 1/tu2m



#-----------------------------------
# Tritium
#-----------------------------------
# Portland
ifile = './Tritium/wiser_gnip-monthly-us-gnipmus02.csv'
trit_atmos = stable_isotope_tools.GNIP('trit_atmos')
trit_atmos.read_gnip_csv(ifile)
trit_atmos.data.rename(columns={'H3':'H3_Portland'}, inplace=True)
trit_portland = trit_atmos.data.H3_Portland

# Ottawa
ifile2 = './Tritium/wiser_gnip-monthly-ca-gnipmca01.csv' 
trit_atmos2 = stable_isotope_tools.GNIP('trit_atmos')
trit_atmos2.read_gnip_csv(ifile2)
trit_atmos2.data.rename(columns={'H3':'H3_Ottowa'}, inplace=True)
trit_ottowa = trit_atmos2.data.H3_Ottowa

# Vienna
ifile3 = './Tritium/wiser_gnip-monthly-at-gnipmat01.csv' 
trit_atmos3 = stable_isotope_tools.GNIP('trit_atmos')
trit_atmos3.read_gnip_csv(ifile3)
trit_atmos3.data.rename(columns={'H3':'H3_Vienna'}, inplace=True)
trit_at = trit_atmos3.data.H3_Vienna

# Michel and Jurgens 2018, DOI: https://doi.org/10.3133/sir20185086
# Interpolated tritium across CONUS -- only goes to 2012
trit_qd = pd.read_csv('./Tritium/TritiumInQuadrangeles_Jurgens2018.csv')
trit_riv = trit_qd.loc[:,['Date', 'Year', 'Month','39-37,110-105']] # these are Crested Butte degree coordinates
trit_riv.columns = ['Date', 'Year', 'Month','H3']
trit_riv.index = pd.to_datetime(trit_riv['Date'])
trit_riv = trit_riv['H3'][:713] # drop the last 3 or so nan values


### Extend Michel forward
# Paper suggests using Vienna and the given equation.
# Vienna stops at 2012 though. Are they going to continue the timseries?

# Make my own correlation against Ottowa
trit_df = pd.merge(trit_ottowa, trit_riv, how='outer', left_index=True, right_index=True)
# Sort the month
trit_df['Month'] = [trit_df.index[i].month for i in range(len(trit_df))]
trit_mn = trit_df.sort_values(by='Month')
trit_mn.dropna(axis=0, how='any', inplace=True)
trit_mn_ar = trit_mn.to_numpy()

# do a monthly linear regression using log TU
def tu_reg(ott_j, rvt_j):
    '''feed in timeseries of a single month'''    
    # predict rvt as function of ott
    slope, intercept, r_value, p_value, std_err = stats.linregress(ott_j, rvt_j)
    return slope, intercept, r_value

ott_rvt_corr = []
for i in range(1,13):
    dm = trit_mn_ar[trit_mn_ar[:,2] == float(i)]
    ott_rvt_corr.append(tu_reg(np.log10(dm[:,0]), np.log10(dm[:,1])))

# make a dataframe
df_corr = pd.DataFrame(data=ott_rvt_corr, index=np.arange(1,13), columns=['slope','intercept','r2'])
df_corr.index.name = 'month'

# extend Riverton tritium forward using the above relationship
trit_df['H3_ext'] = trit_df['H3'].copy()
for i in trit_df.index[trit_df['H3'].isna()]:
    ott = np.log10(trit_df.loc[i,'H3_Ottowa'])
    log_rvt_ext = df_corr.loc[i.month,'slope']*ott + df_corr.loc[i.month,'intercept']
    trit_df.loc[i, 'H3_ext'] = 10**log_rvt_ext
    
# still some straggler nan values, use some interpolation to fill them
# extend the tritium series back in time using background concentrations?
# or take current values to be background?
rvt_trit = trit_df['H3_ext']
rvt_trit.interpolate(method='linear', axis=0, limit=10, inplace=True, limit_direction='forward', limit_area=None)
background = rvt_trit.min()

backtimes = pd.date_range(start='1930-01-15', end='1953-01-15', freq='M')
backtimes = backtimes.map(lambda x: x.replace(day=15))
back_df = pd.Series(data=background*np.ones(len(backtimes)), index=backtimes)
# set TU from 1945 to Nan -- this is when the first bomb was exploded
#back_df['1947-12-15':] = np.NaN
# combine
rvt_trit_ext = pd.concat((back_df, rvt_trit))
# use interpolation to fill nans
rvt_trit_ext.interpolate(method='quadratic', inplace=True)
rvt_trit_ext.name = 'tritium'

### Plot
#fig, ax = plt.subplots()
#ax.semilogy(rvt_trit_ext)
#plt.show()



#--------------------------------------------
# CFC and SF6
#--------------------------------------------
# This is the compiled data from gen_cfc_sf6_ts.py in NOAA_cfc_sf6 dir
cfc_sf6 = pd.read_csv('./NOAA_cfc_sf6/cfc_sf6_atmhist_1765-2021.csv', index_col='year')
cfc_sf6.index = pd.to_datetime(cfc_sf6.index.astype(str) + '6' + '15', format='%Y%m%d')
cfc_sf6.index.name = None



#--------------------------------------------
# Aqueous Concentrations
#--------------------------------------------
# Compile atomospheric cfc, sf6, and tritium into single dataframe
# This performs Nan filling and some sort of interpolation
tracer_atmos = get_atm_conc.concat_resampled_time_series(C_t_gas=cfc_sf6, C_t_isotopes=rvt_trit_ext, freq='M', hemisphere='NH')
tracer_atmos.rename(columns={'cfc11_ppt':'CFC11NH','cfc12_ppt':'CFC12NH','cfc113_ppt':'CFC113NH',
                             'sf6_ppt':'SF6NH',
                             'tritium':'H3'}, inplace=True)

# Resample 6 month averages
tracer_atmos_ = tracer_atmos.resample('Y').mean()


tracer_atmos_.rename(columns={'CFC11NH':'CFC11',
                              'CFC12NH':'CFC12',
                              'CFC113NH':'CFC113',
                              'SF6NH':'SF6'}, inplace=True)

tracer_atmos_.index.name = 'Date'

# Convert atmos mixing ratios to aqueous concentrations
# Only applies to gas special
#T = 4. # Recharge Temp, C
#E = 2900.0 # Recharge Elevation, meters
#P = cfc_tools.lapse_rate(E)
#tracer_aq = get_atm_conc.convert2aqueous(tracer_atmos_,T,P,S=0.,addHe=True,addAr39=True,addKr81=True,hemisphere='NH')
#tracer_aq = tracer_aq.rename(columns = {'H3':'trit TU'})


# Convert to molality
#tracer_molal = get_atm_conc.convert2molality(tracer_aq)
#tracer_molal = tracer_molal[['CFC11','CFC12','CFC113','SF6','H3','He3','He4']]
#tracer_molal['Date'] = tracer_molal.index.date



#---------------------------------
# Save to files
#---------------------------------
#tracer_molal.to_csv('CB_tracer_aq.csv', index=False)
tracer_atmos_.to_csv('CB_tracer_atmos.csv')



#----------------------------
# Plotting
#----------------------------
# Mangle the Units
#tracer_molal.loc[:,'CFC11'] = tracer_molal.loc[:,'CFC11'].astype(float)*1.e12 
#tracer_molal.loc[:,'CFC12'] = tracer_molal.loc[:,'CFC12'].astype(float)*1.e12 
#tracer_molal.loc[:,'CFC113'] = tracer_molal.loc[:,'CFC113'].astype(float)*1.e12 
#tracer_molal.loc[:,'SF6'] = tracer_molal.loc[:,'SF6'].astype(float)*1.e15 

tracer_in = tracer_atmos_.copy()

fig, ax = plt.subplots(figsize=(10,6))
fig.subplots_adjust(left=0.15, right=0.9, top=0.90)
ax.plot(tracer_in['CFC11'], c='C0', linestyle='--', label='CFC-11')
ax.plot(tracer_in['CFC12'], c='C1', linestyle='--', label='CFC-12')
ax.plot(tracer_in['CFC113'], c='C2', linestyle='--', label='CFC-113')
ax.plot(tracer_in['SF6']*10, c='C3', linestyle=':', label=r'$\mathrm{SF_{6}}$')
ax.set_ylabel(r'$\mathrm{CFCs \ (pptv)}$'+'\n'+r'$\mathrm{SF_{6}*10 \ (pptv)}$') #,color='red')
#ax.yaxis.set_minor_locator(MultipleLocator(0.25e-12))
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))

ax1 = ax.twinx()
ax1.plot(tracer_in['H3'], c='C4', linestyle='solid', label=r'$\mathrm{ ^{3}H}$')
ax1.set_yscale('log')
ax1.set_ylabel(r'$\mathrm{^{3}H \ (TU)}$')#, color='C0')
#ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

#ax.xaxis.set_minor_locator(MultipleLocator(365*10))
ax.set_xlabel('Date')
ax.grid()
ax.set_xlim(pd.to_datetime('1949-12-31'), pd.to_datetime('2020-12-31'))

fig.legend(loc='center', ncol=5, bbox_to_anchor=(0.52, 0.94))
plt.savefig('Atmos_tracer_input.jpg',dpi=320, format='jpg')
plt.show()



