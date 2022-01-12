from mcphysics import data
import spinmob as s
import mcphysics
import numpy as np

def gaussian(peak_domain=None, A0=None, b0=None, sigma0=None):
    """ Function to fit Gaussian to compton scattering data. 
    
    Parameters:
    ----------
    peak_domain (optional): list of 2 int, domain of peak under consideration;
    A0 (optional): int, initial guess for parameter A;
    b0 (optional): int, initial guess for parameter b;
    sigma0 (optional): int, initial guess for parameter sigma;
    
    Returns:
    --------
    fit_parameters: array, parameters obtained through fitting [A, A.std, b, b.std, sigma, sigma.std];  
    """

    data_set = mcphysics.data.load_chns() # load .chn data files
    runs = len(data_set) # number of data files loaded
    f = s.data.fitter() # initiate fitter object
    
    # initiate parameter arrays
    A = np.zeros(runs) 
    A_std = np.zeros(runs)
    b = np.zeros(runs)
    b_std = np.zeros(runs)
    sigma = np.zeros(runs)
    sigma_std = np.zeros(runs)

    if peak_domain is not None:
        if not isinstance(peak_domain, list) or len(peak_domain) != 2:
            return TypeError('peak_domain must be a list of size 2.')

    if runs==1: # single run
        data = data_set[0] 
        f.set_functions(f = 'A * exp(-0.5*((x - b)/sigma)**2)/(sigma*sqrt(2*pi))', p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0))
        if peak_domain is None:
            f.set_data(xdata = data['Channel'], ydata = data['Counts'], xlabel='Channel', ylabel='Counts')
        else:
            f.set_data(xdata = data['Channel'][peak_domain[0]:peak_domain[1]], ydata = data['Counts'][peak_domain[0]:peak_domain[1]], xlabel='Channel', ylabel='Counts')
        f.fit() # fit to data
        A = f.get_fit_results()['A']
        A_std = f.get_fit_results()['A.std']
        b = f.get_fit_results()['b']
        b_std = f.get_fit_results()['b.std']
        sigma = f.get_fit_results()['sigma']
        sigma_std = f.get_fit_results()['sigma.std']

    else: # multiple runs
        for i in range(0,runs):
            data = data_set[i]
            f.set_functions(f = 'A * exp(-0.5*((x - b)/sigma)**2)/(sigma*sqrt(2*pi))', p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0))
            if peak_domain is None:
                f.set_data(xdata = data['Channel'], ydata = data['Counts'], xlabel='Channel', ylabel='Counts')
            else:
                print(data)
                f.set_data(xdata = data['Channel'][peak_domain[0]:peak_domain[1]], ydata = data['Counts'][peak_domain[0]:peak_domain[1]], xlabel='Channel', ylabel='Counts')
            f.fit()
            A[i] = f.get_fit_results()['A']
            f.fit() # fit to data
            A_std[i] = f.get_fit_results()['A.std']
            b[i] = f.get_fit_results()['b']
            b_std[i] = f.get_fit_results()['b.std']
            sigma[i] = f.get_fit_results()['sigma']
            sigma_std[i] = f.get_fit_results()['sigma.std']
    fit_parameters = [A, A_std, b, b_std, sigma, sigma_std]
    return fit_parameters

#__________________________________________________________________________________

def calibrate(energy, ch_peaks, ch_unc):

    

    data_set = mcphysics.data.load_chns() # load .chn data files
    runs = len(data_set) # number of data files loaded
    f = s.data.fitter() # initiate fitter object

    # initiate parameter arrays
    m = np.zeros(runs) 
    m_std = np.zeros(runs)
    b = np.zeros(runs)
    b_std = np.zeros(runs)


    f.set_functions(f = 'm*x + b', p = 'm,b')
    f.set_data(xdata=energy, ydata=ch_peaks)
    
#__________________________________________________________________________________

def combine_chns():
    " Select the .chn files to combine into a single spinmob databox. "
    
    databoxes = mcphysics.data.load_chns()
    databox = databoxes[0]
    for box in databoxes:
        databox['Counts'] =+ box['Counts']

    return databox