# NobleGas_RTD_MCMC
This repository contains python scripts to interpret dissolved noble gases and environmental-age tracers using Markov-chain Monte Carlo techniques.

Directory Notes and Procedure

(1) - Field_Data: contains the environmental tracer and dissolved noble gas meaurements
(2) - Input_Series: contains the atmospheric input functions for the environmental tracers
(3) - WeatherStations: calculates air temperature lapse rates using weather station and SNOTEL data
(4) - ng_interp: contains scripts to perform the MCMC analysis on the dissolved noble gas measurements
                 run the noble_gas_mcmc.py to perfrom the inference
(5) - main dir: run age_modeling_mcmc.prep.py to pre-process the environmental tracer field observations with the output from the noble gas analysis.
(6) - age_ens_runs_mcmc: calculates mean ages using mcmc analysis
                         run run_age_mcmc.py to perform analysis

Utility Scripts:
    noble_gas_utils.py: calculates noble gas concentrations using the Closed-Equilibrium model
    cfc_utils.py: calculates aqeuous concentration of cfc's and sf6
    convolution_integral_utils.py: performs forward runs of the convolution integral to be used within lumped parameter modeling.


Steps to Re-Create Thiros et al (2023) WRR Manuscript:
  (1) python noble_gas_mcmc.py
  (2) python age_modeling_mcmc.prep.py
  (3) python run_age_mcmc.py
~                                    
