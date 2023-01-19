# Script will write the field observations to csv file that can be used to import into the PANGA software

import numpy as np
import pandas as pd
import pickle
import sys

sys.path.insert(0, '../../utils')
import noble_gas_utils as ng






#--------------------------------
# Load Noble Gas Obs
#--------------------------------
dd_ = pd.read_excel('../../Field_Data/PLM_NobleGas_2021.xlsx', skiprows=1, index_col='SiteID', nrows=5)
dd  = dd_.copy()[['4He','Ne','Ar','Kr','Xe']]
dd.rename(columns={'4He':'He'}, inplace=True)
dd.dropna(inplace=True)
dd.index.name = ''

obs_dict_all = dd.T.to_dict()

well_elev = {'PLM1':2786.889893,
             'PLM6':2759.569824,
             'PLM7':2782.550049}

err_dict={'He':5, 'Ne':5, 'Ar':5, 'Kr':5, 'Xe':12.1}  # avoid Xe depletion by weighting less

for t in list(err_dict.keys()):
    dd['{}_err'.format(t)] = dd[t]*err_dict[t]/100

dd = dd.copy()[['He','He_err','Ne','Ne_err','Ar','Ar_err','Kr','Kr_err','Xe','Xe_err']]

dd = dd.copy().loc[['PLM1','PLM7','PLM6'],:]
dd.index.name = 'wells'
dd.to_csv('ng_conc_4_panga.csv', index=True)



# Predict concentrations then compare to panga with same parameters
gases = ['He','Ne', 'Ar', 'Kr', 'Xe']
E  = 3000
T  = 5.0
Ae = 0.01
F  = 0.5

# Find solubility equilibrium concentrations
ce_mod = ng.noble_gas_fun(gases=gases, E=E, T=T, Ae=Ae, F=F, P='lapse_rate')
# PANGA wants pressures in atm
P_atm  = ce_mod.P * 1.e9 * 9.8692e-6

















