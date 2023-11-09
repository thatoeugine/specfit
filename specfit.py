import numpy as np
import scipy.stats, scipy

from astropy.table import Table
from astropy.io import ascii
import time

import matplotlib.pyplot as plt
import pymultinest
import json
import pandas as pd
import seaborn as sns
import functiontools

sns.set_style("whitegrid")


import matplotlib.pyplot as plt
plt.rcParams.update({'legend.fontsize': 12,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'axes.linewidth': 0.5,
          'axes.edgecolor': '.15',
          'xtick.labelsize' :15,
          'ytick.labelsize': 15,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'mathtext.bf': 'serif:bold',
          'mathtext.it': 'serif:italic',
          'mathtext.sf': 'sans\\-serif',
         })



datafile = '../subcube_0.5_hatlas1429_track2-fitorder3-chans1-140_270-299.linefile.fits-Z-profile-Region_1-Statistic_FluxDensity-Cooridnate_Current-2023-04-04-16-59-13.tsv'
noisefile = '../noise_values_subcube_0.5_hatlas1429_track2-fitorder3-chans1-140_270-299.linefile.fits-Z-profile-Region_1-Statistic_RMS-Cooridnate_Current-2023-04-04-17-02-19.tsv'
outdir = 'out/' # Directory where the output from PyMultiNest and this script are stored



# Load data
data = ascii.read(datafile, comment = '#')
noisedata  = ascii.read(noisefile)

df = data.to_pandas()
df2 = noisedata.to_pandas()


xdata = df['Frequency']*1e6 #kHz
ydata = df['Flux']*1e3  #mJy
noise = df2['Noise']*1e3 #mJy



# Define function that uses PyMultiNest to fit a model to the data and returns the evidence and median parameters
def fit_gaussian(model_name, param_list, model):
    
    # Get dimension of problem
    n_params = len(param_list)
    
    # Define prior and loglikelihood-------------------------------------------
    
    if model_name == 'A':
        def prior(cube, ndim, nparams):
            """
            The prior transform going from the unit hypercube to the true parameters. This function
            has to be called "Prior".

            Args:
                cube (numpy.ndarray): an array of values drawn from the unit hypercube

            Returns:
                :numpy.ndarray: an array of the transformed parameters
            """
            cube[0] = cube[0] * (824e3 - 822e3) + 822e3 # pos1: [0, 1] -> [822000, 824000]
            cube[1] = cube[1] * (1e3  - 8.301) + 8.301  # fwhm1: [0, 1] -> [8.301, 1000]
            cube[2] = cube[2] * (50  - 1.085) + 1.085  # height1: [0, 1] -> [1.085, 50]

        
    elif model_name == 'B':
        def prior(cube, ndim, nparams):
            cube[0] = cube[0] * (824e3 - 822e3) + 822e3 # pos1: [0, 1] -> [822000, 824000]
            cube[1] = cube[1] * (1e3  - 8.301) + 8.301  # fwhm1: [0, 1] -> [8.301, 1000]
            cube[2] = cube[2] * (50  - 1.085) + 1.085  # height1: [0, 1] -> [1.085, 50]
            cube[3] = cube[3] * (824e3 - 822e3) + 822e3 # pos2: [0, 1] -> [822e3, 824e3]
            cube[4] = cube[4] *  (1e3 - 8.301) + 8.301  # fwhm2: [0, 1] -> [8.301, 1e3]
            cube[5] = cube[5] * (50  - 1.085) + 1.085 # height2: [0, 1] -> [1.085, 50]
            
             # pos1< pos2 < pos3 < pos4 < ....< posN:
            pos1 = cube[0]
            pos2 = cube[3]

            sorted_positions = sorted([pos1, pos2])

            cube[0] = sorted_positions[0]
            cube[3] = sorted_positions[1]

                
    elif model_name == 'C':
        def prior(cube, ndim, nparams):
            cube[0] = cube[0] * (824e3 - 822e3) + 822e3  # pos1
            cube[1] = cube[1] * (1e3 - 8.301) + 8.301  # fwhm1
            cube[2] = cube[2] * (50  - 1.085) + 1.085  # height1
            cube[3] = cube[3] * (824e3 - 822e3) + 822e3  # pos2
            cube[4] = cube[4] * (1e3 - 8.301) + 8.301  # fwhm2
            cube[5] = cube[5] * (50  - 1.085) + 1.085  # height2
            cube[6] = cube[6] * (824e3 - 822e3) + 822e3  # pos3
            cube[7] = cube[7] * (1e3 - 8.301) + 8.301  # fwhm3
            cube[8] = cube[8] * (50  - 1.085) + 1.085  # height3

            pos1 = cube[0]
            pos2 = cube[3]
            pos3 = cube[6]

            sorted_positions = sorted([pos1, pos2, pos3])

            cube[0] = sorted_positions[0]
            cube[3] = sorted_positions[1]
            cube[6] = sorted_positions[2]

                         
    elif model_name == 'D':
        def prior(cube, ndim, nparams):
            cube[0] = cube[0] * (824e3 - 822e3) + 822e3 # pos1
            cube[1] = cube[1] * (1e3  - 8.301) + 8.301  # fwhm1
            cube[2] = cube[2] * (50  - 0.362) + 0.362  # height1 lower-limit set to 1sigma (0.362)
            cube[3] = cube[3] * (824e3 - 822e3) + 822e3 # pos2
            cube[4] = cube[4] *  (200 - 8.301) + 8.301  # fwhm2
            cube[5] = cube[5] * (50  - 1.085) + 1.085  # height2
            cube[6] = cube[6] * (824e3 - 822e3) + 822e3 # pos 3
            cube[7] = cube[7] * (200 - 8.301) + 8.301 # fwhm 3
            cube[8] = cube[8] * (50  - 1.085) + 1.085 # height 3
            cube[9] = cube[9] * (824e3 - 822e3) + 822e3 # pos4
            cube[10] = cube[10] * (1e3 - 1) + 1 # fwhm 4
            cube[11] = cube[11] * (50  - 1.085) + 1.085  # height 4
            
            pos1 = cube[0]
            pos2 = cube[3]
            pos3 = cube[6]
            pos4 = cube[9]

            sorted_positions = sorted([pos1, pos2, pos3, pos4])

            cube[0] = sorted_positions[0]
            cube[3] = sorted_positions[1]
            cube[6] = sorted_positions[2]
            cube[9] = sorted_positions[3]
            
            
    elif model_name == 'E':
        def prior(cube, ndim, nparams):
            cube[0] = cube[0] * (824e3 - 822e3) + 822e3 # pos1
            cube[1] = cube[1] * (300  - 8.301) + 8.301  # fwhm1
            cube[2] = cube[2] * (50  - 0.724) + 0.724  # height1 lower-limit set to 1sigma (0.362) now changed to 2sigma
            cube[3] = cube[3] * (824e3 - 822e3) + 822e3 # pos2
            cube[4] = cube[4] *  (1.5e3 - 8.301) + 8.301  # fwhm2
            cube[5] = cube[5] * (5  - 1.085) + 1.085 # height2
            cube[6] = cube[6] * (824e3 - 822e3) + 822e3 # pos 3
            cube[7] = cube[7] * (170 - 8.301) + 8.301 # fwhm 3
            cube[8] = cube[8] * (50  - 1.085) + 1.085 # height 3
            cube[9] = cube[9] * (824e3 - 822e3) + 822e3 # pos4
            cube[10] = cube[10] * (200 - 8.301) + 8.301 # fwhm 4
            cube[11] = cube[11] * (50  - 1.085) + 1.085  # height 4
            cube[12] = cube[12] * (824e3 - 822e3) + 822e3 # pos5
            cube[13] = cube[13] *  (50 - 4) + 4 # fwhm 5
            cube[14] = cube[14] * (50  - 1.085) + 1.085  # height 5


            pos1 = cube[0]
            pos2 = cube[3]
            pos3 = cube[6]
            pos4 = cube[9]
            pos5 = cube[12]


            sorted_positions = sorted([pos1, pos2, pos3, pos4, pos5])

            cube[0] = sorted_positions[0]
            cube[3] = sorted_positions[1]
            cube[6] = sorted_positions[2]
            cube[9] = sorted_positions[3]
            cube[12] = sorted_positions[4]

            
    elif model_name == 'F':
        def prior(cube, ndim, nparams):
            cube[0] = cube[0] * (824e3 - 822e3) + 822e3 # pos1
            cube[1] = cube[1] * (300  - 8.301) + 8.301  # fwhm1
            cube[2] = cube[2] * (50  - 0.724) + 0.724  # height1 lower-limit set to 1sigma (0.362) now changed to 2sigma
            cube[3] = cube[3] * (824e3 - 822e3) + 822e3 # pos2
            cube[4] = cube[4] *  (1.5e3 - 8.301) + 8.301  # fwhm2
            cube[5] = cube[5] * (5  - 1.085) + 1.085 # height2
            cube[6] = cube[6] * (824e3 - 822e3) + 822e3 # pos 3
            cube[7] = cube[7] * (170 - 8.301) + 8.301 # fwhm 3
            cube[8] = cube[8] * (50  - 1.085) + 1.085 # height 3
            cube[9] = cube[9] * (824e3 - 822e3) + 822e3 # pos4
            cube[10] = cube[10] * (200 - 8.301) + 8.301 # fwhm 4
            cube[11] = cube[11] * (50  - 1.085) + 1.085  # height 4
            cube[12] = cube[12] * (824e3 - 822e3) + 822e3 # pos5
            cube[13] = cube[13] *  (50 - 4) + 4 # fwhm 5
            cube[14] = cube[14] * (50  - 1.085) + 1.085  # height 5
            cube[15] = cube[15] * (824e3 - 822e3) + 822e3 # pos6
            cube[16] = cube[16] *  (400 - 8.301) + 8.301 # fwhm 6 
            cube[17] = cube[17] * (5  - 1.085) + 1.085  # height 6

            
            pos1 = cube[0]
            pos2 = cube[3]
            pos3 = cube[6]
            pos4 = cube[9]
            pos5 = cube[12]
            pos6 = cube[15]

            sorted_positions = sorted([pos1, pos2, pos3, pos4, pos5, pos6])

            cube[0] = sorted_positions[0]
            cube[3] = sorted_positions[1]
            cube[6] = sorted_positions[2]
            cube[9] = sorted_positions[3]
            cube[12] = sorted_positions[4]
            cube[15] = sorted_positions[5]


    def loglike(cube, ndim, nparams):
        params = [cube[i] for i in range(ndim)]
        ymodel = model(xdata, *params)
        
        loglikelihood = (-0.5 * ((ymodel - ydata) / noise)**2).sum()
        return loglikelihood
    
    
    # Run multinest------------------------------------------------------------
    output_prefix = outdir + model_name + '_'
    start_time = time.time()
    pymultinest.run(loglike, prior, n_params, outputfiles_basename=output_prefix, resume = True, verbose = False)
    runtime = time.time() - start_time
    json.dump(param_list, open(output_prefix + 'params.json', 'w')) # save parameter names
    print('\n\nFinished running MulitNest for '+model_name)

    # Get analyzer object from PyMultiNest
    a = pymultinest.Analyzer(outputfiles_basename=output_prefix, n_params = n_params)
    
    # Get bayesian evidence
    s = a.get_stats()
    evidence = s['global evidence'] # the natural log of the evidence is returned
    
    # Get median parameter values
    med_params = []
    med_params_sigma = []
    for p, m in zip(param_list, s['marginals']):
        med = m['median']
        med_sigma = m['sigma']
        med_params.append(med)
        med_params_sigma.append(med_sigma)
        
    # Produce corner plot
    print('\nMaking corner plots.')
    functiontools.plot_corner(a, param_list, output_prefix)
    
    return (evidence, med_params, med_params_sigma, runtime)


#------------------------------------------------------------------------------
# Fit the data to each of the above models in turn and store the evidence and
# the median model parameters
#------------------------------------------------------------------------------
model_names = ['A', 'B', 'C', 'D', 'E', 'F']#, 'G', 'H', 'I', 'J', 'K', 'L']
model_list = functiontools.multiple_peaks

param_list = ['pos1', 'fwhm1', 'height1', 'pos2', 'fwhm2', 'height2',
              'pos3', 'fwhm3', 'height3', 'pos4', 'fwhm4', 'height4',
             'pos5', 'fwhm5', 'height5', 'pos6', 'fwhm6', 'height6',
             'pos7', 'fwhm7', 'height7', 'pos8', 'fwhm8', 'height8',
            'pos9', 'fwhm9', 'height9', 'pos10', 'fwhm10', 'height10'
            'pos11', 'fwhm11', 'height11', 'pos12', 'fwhm12', 'height12']


evidence = []
median_params = []
median_params_sigma = []
runtime = []


for i in range(len(model_names)):
    e, mp, mp_sigma, rnt = fit_gaussian(model_names[i], param_list[0:3+i*3], model_list)
    print(i)
    evidence.append(e)
    median_params.append(mp)
    median_params_sigma.append(mp_sigma)
    runtime.append(rnt)
    
    output_prefix = outdir + model_names[i] + '_'

    data_frameDic = {'med_params':median_params[i],
                     'med_params_err': median_params_sigma[i], 
                    'runtime': runtime[i]}


    df = pd.DataFrame(data_frameDic)
    df.to_csv(output_prefix + 'params.csv', index = False)  
    
    
    #------------------------------------------------------------------------------
    # Plot median models
    #------------------------------------------------------------------------------

    fig,(ax1, ax2) = plt.subplots(2, sharex=True, figsize=(15,9), gridspec_kw={'height_ratios':[4,1]})
    ax1.errorbar(x=xdata, y=ydata, yerr=noise, ecolor='#95a5a6', fmt='k.', lw= 1, capsize=3,
                 capthick=2, label='Data')

    ax1.plot(xdata, functiontools.multiple_peaks(xdata, *median_params[i]),
             ls='-', lw=1.8, color='tab:blue', label='Model'+' '+model_names[i])
    ax2.plot(xdata, ydata - functiontools.multiple_peaks(xdata, *median_params[i]), color = 'tab:blue')


    plt.xlabel(r'Frequency (kHz)')
    ax1.set_ylabel(r'Flux density (mJy/beam)')
    ax1.legend(fontsize=14)


    ax2.set_ylabel(r'Residuals')
    ax2.hlines(0, xdata.min(), xdata.max(), color= 'black')
    
    plt.tight_layout()
    plt.savefig(output_prefix+'median_fits'+'.pdf', dpi=600)
    plt.show()
    plt.close()
