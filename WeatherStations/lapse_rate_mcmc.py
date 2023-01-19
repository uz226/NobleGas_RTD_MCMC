import pandas as pd
import numpy as np
from scipy import stats
import calendar

import pdb

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
plt.rcParams['font.size']=14

import matplotlib.style as style
style.use('tableau-colorblind10')


#--------------
# Data imports
#--------------
# import the weather station/snotel lat long info
coords = pd.read_excel('WS_lat_long.xlsx', index_col='Location')
coords['Elevation_m'] = coords['Elevation_ft'].copy()/3.28084


# import raw weather station data
dd1 = pd.read_csv('ws_airtemp.csv')
# cleanup
dd1 = dd1[['Location',' Timestamp','Value']].rename(columns={' Timestamp':'Timestamp'})
dd1.index = dd1['Location']
dd1.drop(columns='Location', inplace=True)
dd1.index = [i.split()[0][3:] for i in dd1.index]
dd1['Timestamp'] = pd.to_datetime(dd1['Timestamp'])


# import raw stotel data
dd2 = pd.read_csv('snotel_airtemp.csv')
# cleanup
dd2 = dd2[['Location',' Timestamp','Value']].rename(columns={' Timestamp':'Timestamp'})
dd2.index = dd2['Location']
dd2.drop(columns='Location', inplace=True)
dd2.index = [i.split()[0][3:] for i in dd2.index]
dd2['Timestamp'] = pd.to_datetime(dd2['Timestamp'])


#-----------
# Drop data
#-----------
# Mexican Cut does not look good. Drop it
#coords.drop(index=['Mexican_Cut'], inplace=True)
#dd1.drop(index=['Mexican_Cut'], inplace=True)

# testing without almont, which shows high variance
#coords.drop(index=['Almont'], inplace=True)
#dd1.drop(index=['Almont'], inplace=True)


# combine all data
dd = pd.concat((dd1,dd2))

# add in station elevation
for i in range(len(coords)):
    ee = (coords.iloc[i,-1])
    nn = coords.index[i]
    if nn in dd.index.unique():
        print ('found {}'.format(nn))
        dd.loc[coords.index[i], 'Elevation'] = ee
dd.dropna(how='any', inplace=True)
   

#-------
#QA/QC plots
# combined data
fig, ax = plt.subplots(figsize=(10,8))
for i in range(len(dd.index.unique())):
    station_inds = dd.index == dd.index.unique()[i]
    ax.scatter(dd.loc[station_inds]['Timestamp'], dd.loc[station_inds]['Value'], marker='.', s=5.0, label=dd.index.unique()[i])
    #ax.scatter(dd['Value'], dd['Elevation'])
ax.set_ylabel('Temperature (C)')
ax.set_xlabel('Date')
ax.legend()
plt.show()


# seperate subplots
fig, ax = plt.subplots(ncols=2, nrows=np.ceil(len(dd.index.unique())/2).astype(int), figsize=(10,10))
for i in range(len(dd.index.unique())):
    r = i//2
    c = i%2
    station_inds = dd.index == dd.index.unique()[i]
    ax[r,c].scatter(dd.loc[station_inds]['Timestamp'], dd.loc[station_inds]['Value'], marker='.', s=5.0, label=dd.index.unique()[i])
    ax[r,c].set_title(dd.index.unique()[i])
fig.tight_layout()
plt.show()



#---------------
# Data clipping
#---------------
yrs_to_use = [2018, 2019]
keep_inds = []
for t in dd['Timestamp']:
    if t.year in yrs_to_use:
        keep_inds.append(True)
    else:
        keep_inds.append(False)

dd = dd[keep_inds]


# Replot
fig, ax = plt.subplots(ncols=2, nrows=np.ceil(len(dd.index.unique())/2).astype(int), figsize=(10,10))
for i in range(len(dd.index.unique())):
    r = i//2
    c = i%2
    station_inds = dd.index == dd.index.unique()[i]
    temps = dd.loc[station_inds]['Value']
    ax[r,c].scatter(dd.loc[station_inds]['Timestamp'], temps, marker='.', s=5.0, label=dd.index.unique()[i])
    # add average line
    ax[r,c].axhline(temps.mean(), linestyle='--', color='grey')
    # cleanup
    ax[r,c].set_title(dd.index.unique()[i])
    ax[r,c].xaxis.set_major_locator(YearLocator())
    ax[r,c].xaxis.set_major_formatter(DateFormatter('%Y'))
    ax[r,c].set_ylim(-25, 25)
    ax[r,c].yaxis.set_major_locator(MultipleLocator(10))
    ax[r,c].yaxis.set_minor_locator(MultipleLocator(5))
ax[2,0].set_ylabel('Temperature (C)')
fig.tight_layout()
plt.show()






#---------
# Montly analysis
#---------
# monthly average dataframe
df_month = []
hh = []
ee = []
for i in range(len(dd.index.unique())):
    station_inds = dd.index == dd.index.unique()[i]
    month_avg = dd[station_inds].groupby(pd.Grouper(key='Timestamp',freq='1M')).mean()
    df_month.append(month_avg['Value'])
    hh.append(dd.index.unique()[i])
    ee.append(coords.loc[dd.index.unique()[i], 'Elevation_m'])
df_month = pd.concat((df_month),axis=1)
df_month.columns = hh


# find equation for best fit line
m = []
intr = []
for i in range(len(df_month)):
    x = df_month.iloc[i,:].to_numpy()
    y = np.array(ee)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    m.append(slope)
    intr.append(intercept)


# Plots for each month
fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(12,10))
fig.subplots_adjust(wspace=0.3,hspace=0.3, left=0.08, right=0.98, top=0.95)
mnt_labs = [calendar.month_abbr[i] for i in range(1,13)]
for i in range(24):
    r,c = (i%12)//4, (i%12)%4
    # plot temp versus elevation
    ax[r,c].scatter(df_month.iloc[i,:], ee)
    ax[r,c].set_title('{}'.format(mnt_labs[i%12]))
    lm = ax[r,c].get_xlim()
    # some best fit lines
    xx = np.arange(-40, 40)
    yy = m[i]*xx + intr[i]
    ax[r,c].plot(xx,yy, linestyle='--', label=df_month.index[i].year)
    if i < 12:
        ax[r,c].text(0.8, 0.9, 'm={:.1f}\nb={:.1f}'.format(m[i],intr[i]), horizontalalignment='center', verticalalignment='center', 
                     transform=ax[r,c].transAxes, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
    else:
        ax[r,c].text(0.8, 0.7, 'm={:.1f}\nb={:.1f}'.format(m[i],intr[i]), horizontalalignment='center', verticalalignment='center', 
                     transform=ax[r,c].transAxes, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
    # clean up
    ax[r,c].set_xlim(lm[0]-2, lm[1]+2)
    ax[r,c].set_ylim(min(ee)*0.95, max(ee)*1.05)
#ax[1,0].set_ylabel('Elevation (m)')
ax[0,0].legend()
fig.text(0.5, 0.04, 'Temperature (C)', ha='center')
fig.text(0.01, 0.5, 'Elevation (m)', va='center', rotation='vertical')
plt.show()




#------
# Yearly Mean Ana
# Treat each year seperate

# test dropping january and december
#months_not = [12, 1]
#keep_inds = []
#for t in dd['Timestamp']:
#    if t.month in months_not:
#        keep_inds.append(False)
#    else:
#        keep_inds.append(True)
#dd = dd[keep_inds]



# Summarize yearly temperatures and take stats
df_yr = []
hh = []
ee = []
#pdb.set_trace()
for i in range(len(dd.index.unique())):
    station_inds = dd.index == dd.index.unique()[i]
    yr_avg = dd[station_inds].groupby(pd.Grouper(key='Timestamp',freq='1Y')).mean()
    df_yr.append(yr_avg['Value'])
    hh.append(dd.index.unique()[i])
    ee.append(coords.loc[dd.index.unique()[i], 'Elevation_m'])
df_yr = pd.concat((df_yr),axis=1)
df_yr.columns = hh


# find equation of best-fit line
m = []
intr = []
rval = []
for i in range(len(df_yr)):
    x = df_yr.iloc[i,:].to_numpy()
    y = np.array(ee)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    m.append(slope)
    intr.append(intercept)
    rval.append(r_value)


#-----
# Combine all years into single statistic
# combine the dataframes
x = df_yr.to_numpy().ravel()
#x = df_yr.to_numpy().ravel() - 1
y = np.array(ee*len(df_yr.index.year))
m, intr, r_value, p_value, std_err = stats.linregress(x,y)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))
xlm = []
for i in range(len(df_yr)):
    # plot temp versus elevation
    ax.scatter(df_yr.iloc[i,:], ee)
# some best fit lines
xx = np.arange(-20, 40)
yy = m*xx + intr
ax.plot(xx,yy, color='black', linestyle='--')
ax.text(0.7, 0.9, 'slope={:.2f}\nintercept={:.2f}\nR2={:.2f}'.format(m,intr,r_value**2), horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='grey', alpha=0.5))
# clean up
ax.set_xlim(x.min()-1, x.max()+x.max()*0.1)
ax.set_ylim(min(ee)*0.95, max(ee)*1.05)
ax.set_xlabel('Air Temperature (C)')
ax.set_ylabel('Elevation (m)')
# add some bands
xbd = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20)
ax.fill_between(x=xbd, y1=np.ones(len(xbd))*2750, y2=np.ones(len(xbd))*2790, color='grey', alpha=0.3)
ax.fill_between(x=xbd, y1=np.ones(len(xbd))*2790, y2=np.ones(len(xbd))*2935, color='grey', alpha=0.3)
plt.show()






#----------
# MCMC
#----------
import pymc3 as pm
import seaborn as sb
import theano
import theano.tensor as tt
#from theano import as_op
from theano.compile.ops import as_op
import os
import arviz as az
from pymc3 import plot_posterior_predictive_glm



with pm.Model() as mod:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    s = pm.HalfCauchy("y_sigma", beta=10, testval=1.0)
    m = pm.Normal("slope", mu=-153.0, sigma=20)
    b = pm.Normal("intercept", mu=3000.0, sigma=600)

    # Define likelihood
    likelihood = pm.Normal("y", mu = b+m*x, sigma=s, observed=y)
    #likelihood = pm.Normal("y", mu = b+m*x, sigma=y*0.05, observed=y)

    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    trace = pm.sample(tune=1000, draws=10000, return_inferencedata=True)

    post_pred = pm.sample_posterior_predictive(trace, samples=1000, random_seed=123423)


trace_summary = az.summary(trace, round_to=6)
print (trace_summary)

# traceplot
az.plot_trace(trace)
plt.show()



# simulate ensemble of lines
xx = np.linspace(x.min()-0.5, x.max()+0.5, 500)
m_bf = trace_summary.loc['slope','mean']
m_sd = trace_summary.loc['slope','sd']*1
b_bf = trace_summary.loc['intercept','mean']
b_sd = trace_summary.loc['intercept','sd']*1
y_bf = m_bf*xx + b_bf



# Plot
fig, ax = plt.subplots(figsize=(5,3.5))
#fig.subplots_adjust(top=0.98, bottom=0.15, left=0.15)
ax.plot(xx, y_bf, color='black', linewidth=4, zorder=9, label='best-fit line')
# simulate 100 new lines that are within 1 sigma
mrand = np.random.normal(m_bf, m_sd, 100)
brand = np.random.normal(b_bf, b_sd, 100)
for i in range(len(mrand)):
    ax.plot(xx, mrand[i]*xx + brand[i], color='grey', alpha=0.4)

# Add in observations
ax.scatter(x, y, zorder=10, facecolors='none', edgecolors='black')
ax.set_xlabel('Air Temperature (C)')
ax.set_ylabel('Elevation (m)')

ax.text(0.62, 0.86, '$\it{{m}}$={:.0f} ({:.0f})\n$\it{{b}}$={:.0f} ({:.0f})'.format(m_bf, m_sd, b_bf, b_sd), horizontalalignment='left', verticalalignment='center', 
        transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.yaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.margins(x=0.01)
ax.grid()
fig.tight_layout()
plt.savefig('lapse_rate_mcmc.png', dpi=300)
plt.savefig('lapse_rate_mcmc.svg', format='svg')
plt.show()















