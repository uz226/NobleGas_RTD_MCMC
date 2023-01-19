# Script to make PFLOTRAN style input for tracer transport simulations

import numpy as np
import pandas as pd
import os
import sys
import datetime
import pdb
import matplotlib.pyplot as plt

def NOAA_dataframe(fname, cfc, sf6):
    '''Reads in a NOAA tracer text file and creates a pandas dataframe.'''

    # Read in the data
    f = open(fname, 'r')
    ll = f.readlines()

    # Store the data
    dd = []
    for i in range(len(ll)):
        if ll[i][0] != '#':
            dd.append(ll[i].split())
    f.close()

    # Convert to pandas dataframe
    if cfc:
        df = pd.DataFrame(data=dd[1:], columns=dd[0])
        # Rename columns to generic labels
        if df.shape[1] == 5:
            df.columns = ['yr', 'mon', 'avg_ppt', 'sd', 'N']
        else:
            print 'Do not recognize the number of columns'
    if sf6:
        df = pd.DataFrame(data=dd, columns=['Site', 'yr', 'mon', 'avg_ppt'])

    # Drop nan values
    df_nonan = df.replace(['nan'], [np.nan])
    df_nonan = df_nonan.dropna(axis=0, how='any')

    # Index by Year+month
    df_nonan['date'] = df_nonan.yr.astype(float) + df_nonan.mon.astype(float)/12.
    df_nonan.index = df_nonan['date']

    return df_nonan

def merge_NOAA_dataframes(df1, df2):
    '''This function takes two dataframes for the same tracer and merges them together using dates'''

    # Stack data frames
    df_stack = pd.concat([df1, df2])
    #df_stack.reset_index(drop=True,)
   
    # Sort by date
    df_stack.sort_index(axis=0, inplace=True)
 
    # Drop duplicates
    df_stack = df_stack.loc[~df_stack.index.duplicated(keep='first')]
 
    # Delete replicate columns
    #df_stack.drop(['yr','mon','N','sd'], axis=1, inplace=True)
    
    return df_stack 


### CFC Data
# CFC-11
cfc11_a = NOAA_dataframe('nwr_F11_MM.dat', cfc=True, sf6=False)
cfc11_b = NOAA_dataframe('nwr_F11_MM_2.dat', cfc=True, sf6=False)
cfc11 = merge_NOAA_dataframes(cfc11_a, cfc11_b) # Merge the two dataframes
cfc11.to_csv('NOAA_cfc11.txt', index=False)

# CFC-12
cfc12_a = NOAA_dataframe('nwr_F12_MM.dat', cfc=True, sf6=False)
cfc12_b = NOAA_dataframe('nwr_F12_MM_2.dat', cfc=True, sf6=False)
cfc12 = merge_NOAA_dataframes(cfc12_a, cfc12_b) # Merge the two dataframes
cfc12.to_csv('NOAA_cfc12.txt', index=False)

# CFC-113
cfc113 = NOAA_dataframe('nwr_F113_MM.dat', cfc=True, sf6=False)
cfc113.to_csv('NOAA_cfc13.txt', index=False)



### SF6 Data
sf6 = NOAA_dataframe('sf6_nwr_surface-flask_1_ccgg_month.txt',cfc=False, sf6=True)
sf6.to_csv('NOAA_sf6.txt', index=False)


"""
class pflotran_NOAA_inputdeck(object):
    def __init__(self):
        self.tracer_dfs = [] # pandas dataframe with concentration and date
        #self.tracer_list = [] # Name of tracers
        
        self.tracer_conc_combine = [] # Combined tracers
        self.tracer_conc_combine_nonan = [] # Combined tracers with no Nan types
        
    def add_tracer_df(self, tracer_dfs, tracer_name):
        '''Add tracer dataframes to a master list'''
        # Drop some of the columns first, just to clean it up
        tr = tracer_dfs.drop(['yr','mon','N','sd', 'date'], axis=1)
        tr.columns = ['avg_'+str(tracer_name)]
        return self.tracer_dfs.append(tr)
    
    def join_tracer_dataframes(self):
        '''Take the seperate tracer dataframes and merge them
           Make sure to add tracers in same order as the tracer list
           This will combine all tracers that utilized 'add_tracer_df' function'''
        
        # Combine the dataframes
        self.tracer_conc_combine = self.tracer_dfs[0].join(self.tracer_dfs[1:])
        
        # Fill the Nan values with the nearest interpolation
        trc = []
        for i in range(self.tracer_conc_combine.shape[1]):
            trc.append(self.tracer_conc_combine.iloc[:,i].interpolate(method='nearest').ffill().bfill())
        self.tracer_conc_combine_nonan = pd.DataFrame(trc).transpose()
        return 
    
    def build_pflo_input(self, filesavename):
        '''Write a new PFLOTRAN tracer input card
           tracers_df is the dataframe with no Nan values
           filesavename is what you want the file to be named'''
           
        # Change the format of the date
        date = self.tracer_conc_combine_nonan.index.strftime('%m_%Y')
        
        # Pull the name of the tracers
        tracer_names = []
        for i in self.tracer_conc_combine_nonan.columns:
            tracer_names.append(i[4:])
        
        # Write a new file
        f = open(str(filesavename), 'w')
        for i in range(len(date)):
            head = 'CONSTRAINT %s\n  CONCENTRATIONS\n' % (date[i])
            f.writelines(head)
            for j in range(len(tracer_names)):
                tr = '    %s  %s  T\n' % (tracer_names[j], self.tracer_conc_combine_nonan.iloc[i,j])
                f.writelines(tr)
            f.writelines('  /\nEND\n')
        f.close()
        
        return 

###
# Build the PFLOTRAN style input card
# If tracer measurements have different temporal resolution
# the longest series will be kept and shorter series will be augemented
###
pf = pflotran_NOAA_inputdeck()
pf.add_tracer_df(cfc11, 'cfc11')
pf.add_tracer_df(cfc12, 'cfc12')
pf.add_tracer_df(cfc113, 'cfc113')
pf.join_tracer_dataframes() # This will join all that were added using 'add_tracer_df'
pf.build_pflo_input('NOAA_atmos_input.exfile')
"""