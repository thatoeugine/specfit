import numpy as np
import matplotlib.pyplot as plt
import corner

import seaborn as sns

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





# Define fitting model
def multiple_peaks(x, *params):
    n_peaks = len(params) // 3
    y = 0
    for i in range(n_peaks):
        pos = params[i*3]
        fwhm = params[i*3 + 1]
        height = params[i*3 + 2]
        y += height * np.exp(-(x - pos)**2 / (2 * fwhm**2))
    return y




def plot_corner(pmn_analyzer_object, parameters, filename_prefix):
    # Plot the posterior distributions. Code taken from multinest_marginals_corner.py
    # available on the PyMultiNest GitHub.
    
    data = pmn_analyzer_object.get_data()[:,2:]
    weights = pmn_analyzer_object.get_data()[:,0]

    mask = weights.cumsum() > 1e-5
    mask = weights > 1e-4
    try:

        corner.corner(data[mask,:], weights=weights[mask], labels=parameters,
                      truths=None, show_titles=True, label_kwargs={"fontsize": 13}, 
                      title_kwargs={"fontsize": 13})
    except Exception:
        print('Proceed plotting')
    plt.savefig(filename_prefix + 'corner_'+'.pdf')
    plt.close()
    
    return


def prior(cube, ndim, nparams):
    """
    The prior transform going from the unit hypercube to the true parameters. This function
    has to be called "Prior".

    Args:
        cube (numpy.ndarray): an array of values drawn from the unit hypercube

    Returns:
        :numpy.ndarray: an array of the transformed parameters
    """
    parameters = ['pos', 'fwhm', 'height']
    num_params = nparams // len(parameters)

    positions = []
    for i in range(num_params):
        positions.append(cube[i] * (824e3 - 822e3) + 822e3)

    sorted_positions = sorted(positions)

    for i in range(num_params):
        cube[i] = sorted_positions[i]
