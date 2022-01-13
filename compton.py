import spinmob as s
import mcphysics
import numpy as np

peaks = {'Ba': [[50, 200], [200, 300], [800, 1050]], 'Co': [300, 500], 'Cs': [1500, 2000], 'Na': [1000, 1800]}
energy_dict = {'Ba': [31, 81, 356], 'Co': 122, 'Cs': 661, 'Na': 511}
gaussian_guesses = {'Ba': [{'A0': 7500, 'b0': 100, 'sigma0': 10, 'C0': 10}, {'A0': 3500, 'b0': 250, 'sigma0': 20, 'C0': 10}, {'A0': 3000, 'b0': 950, 'sigma0': 40, 'C0': 10}], 'Co': {'A0': 72000, 'b0': 350, 'sigma0': 20, 'C0': 10}, 'Cs': {'A0': 1200, 'b0': 1700, 'sigma0': 60, 'C0': 10}, 'Na': {'A0': 2500, 'b0': 1400, 'sigma0': 60, 'C0': 10}}

def get_peak_domains():
    """ Retrieve stored peak channel domains."""
    return peaks
#__________________________________________________________________________________

def get_energies():
    """ Retrieve peak energies of different elements """
    return energy_dict
#__________________________________________________________________________________

def gaussian_fit(data, region, A0=None, b0=None, sigma0=None, C0 = None):
    """ Function to fit Gaussian to compton scattering data. 
    
    Parameters:
    ----------
    data:               spinmob databox
    regions (optional): array of ints, domain of peak under consideration;
    A0 (optional):      int, initial guess for parameter A;
    b0 (optional):      int, initial guess for parameter b;
    sigma0 (optional):  int, initial guess for parameter sigma;
    
    Returns:
    --------
    param:              array of the gaussian fit parameters and uncertainties;  
    """
    f = s.data.fitter() # initiate fitter object
    f.set_functions(f = 'A * exp(-0.5*((x - b)/sigma)**2)/(sigma*sqrt(2*pi)) + C', p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+', C='+str(C0))
    if region is None:
        f.set_data(xdata = data['Channel'], ydata = data['Counts'], xlabel='Channel', ylabel='Counts')
    else:
        f.set_data(xdata = data['Channel'][region[0]:region[1]], ydata = data['Counts'][region[0]:region[1]], xlabel='Channel', ylabel='Counts')
    f.fit() # fit to data
    A = f.get_fit_results()['A']
    A_std = f.get_fit_results()['A.std']
    b = f.get_fit_results()['b']
    b_std = f.get_fit_results()['b.std']
    sigma = f.get_fit_results()['sigma']
    sigma_std = f.get_fit_results()['sigma.std']

    param = np.array([A, A_std, b, b_std, sigma, sigma_std])
    
    return param
#__________________________________________________________________________________

def linear_fit(channel, channel_unc, energy):
    """ Fit a reciprocal linear function to data with uncertainties.

    Arguments
    ---------
    channel (_unc): list, domain of the peak channels and their uncertainties;
    energy :        list, the values of the associated energies;
    
    Return
    ------
    param:          array of the linear fit parameters and uncertainties;
    """
    f = s.data.fitter() # create a fitter object
    f.set_functions(f = 'x/m - b/m', p = 'm, b') # set the function as the reciprocal of a linear function
    f.set_data(xdata = energy, ydata = channel, eydata = channel_unc, xlabel='Energy (KeV)', ylabel='Channels') # supply the data
    f.fit() # make the fit
    # Save fit parameters and uncertainties
    param = np.array([f.get_fit_results()['m'], f.get_fit_results()['m.std'], f.get_fit_results()['b'], f.get_fit_results()['b.std']])

    return param
#__________________________________________________________________________________

def combine_chns():
    """ Select the .chn files to combine into a single spinmob databox. 
    
    No arguments.

    Return
    ------
    databox: spinmob databox of the combined counts at each channel;
    """
    databoxes = mcphysics.data.load_chns() # load chns into a list
    databox = databoxes[0] # save first databox 
    databox['Counts'] =- databox['Counts'] # delete counts of saved databox
    for box in databoxes:
        databox['Counts'] =+ box['Counts'] # add all counts to saved databox

    return databox
#__________________________________________________________________________________

def calibrate(n):
    """ Find the linear fit parameters associated to the relationship between peak channel and energy by first finding the gaussian parameters associated to each peak.
    
    Argument
    --------
    n: number of different scatterers (e.g. Ba, Cs, Na);

    Return
    ------
    param: array of linear fit parameters;
    """
    channel = np.zeros(0) # initiate peak channel array
    channel_unc = np.zeros(0) # initiate peak channel unc array
    energy = np.zeros(0) # initiate energy array
    print('Select all files associated to each data point.')
    for i in range(0,n):
        data = combine_chns() # store combined databox 
        element = data.headers['description'][0:2]
        if element == 'Ba':
            for j in range(0,3): # loop over all peaks of Barium
                _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, peaks[element][j], **gaussian_guesses[element][j]) # fit gaussian 
                energy = np.append(energy, energy_dict[element][j])
                channel = np.append(channel, b)
                channel_unc = np.append(channel_unc, b_std + sigma + sigma_std) # width of peak
        else:
            _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, peaks[element], **gaussian_guesses[element]) # fit gaussian 
            energy = np.append(energy, energy_dict[element])
            channel = np.append(channel, b)
            channel_unc = np.append(channel_unc, b_std + sigma + sigma_std) # width of peak
        
    param = linear_fit(channel, channel_unc, energy) # do a linear fit for the channels computed against energy
    
    return param
#__________________________________________________________________________________