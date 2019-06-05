# Readme for Ch3

All python code in this folder was written and run with python 2.7

The files provide the functions the are necessary to obtain the results that are shown in chapter 2 of the dissertation.
The source code for the related article [Herfurth, Tim, and Tatjana Tchumatchenko. "Information transmission of mean and variance coding in integrate-and-fire neurons." Physical Review E 99.3 (2019): 032420.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.032420) is provided [here](https://github.com/t8ch/dissertation-code/tree/master/Ch3/source_code_PhysRevE.99.032420). It includes the code for the generation of data points and figures, and an instructions for running the code.

## Files and their function
- [AmAutoAll.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/AmAutoAll.py): autocorrelation function based on convolution theorem
- [corr_fun.pyx](https://github.com/t8ch/dissertation-code/blob/master/Ch3/corr_fun.pyx): cross-correlations based on convolution theorem (in cython for better performance)
- [custom_format.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/custom_format.py): customize matplotlib axis formatter
- [davies_harte.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/davis_harte.py): simulation of Gaussian processes (OUP, white noise, stochastic oscillations) with algorithm of Davies & Harte
- [fm.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/fm.py): generation of mean and variance modulated input currents
- [gaussianity_check.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/gaussianity_check.py): checking normality, correlatedness, and multivariate Gaussianity (of simulated spike trains; references for code therein)
- [hypergeometric.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/AmAutoAll.py): (hypergeometric) helper functions for analytic calculations in linear response theory
- [MM_AM_comb.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/MM_AM_Comb.py): computing correlation functions and information for combined encoding (simulation and analytic)
- [MM_AM_mono.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/MM_AM_mono.py): computing correlation functions and information for MM and VM (simulation and analytic
- [signalsmooth.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/signalsmooth.py): smoothing time series with different window (based on: http://scipy.org/Cookbook/SignalSmooth)
- [style_sheet.py](https://github.com/t8ch/dissertation-code/blob/master/Ch3/style_sheet.py): customize plot styles
- [timecourse.pyx](https://github.com/t8ch/dissertation-code/blob/master/Ch3/timecourse.pyx): computing voltage time evolution and spike times for LIF and EIf neurons (implementation of solving differential equation for given input current via Euler forward); implemented in cython for better performance
