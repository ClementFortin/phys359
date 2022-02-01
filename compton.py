import spinmob as s
import mcphysics
import numpy as np

# n is the number of batch of 10-files calibration data
n = 12
peaks = {'Ba': [[75, 130], [220, 280], [700, 885], [900, 1050]], 'Co': [335, 450], 'Cs': [1600, 1900], 'Na': [1225, 1525]}
energy_dict = {'Ba': [30.973, 80.9979, 302.8508, 356.0129], 'Co': [122.06065, 136.47356], 'Cs': 661.657, 'Na': 511.0}
gaussian_guesses = {'Ba': [{'A0': n*75000, 'b0': 100, 'sigma0': 10, 'C0': n*100}, {'A0': 2*35000, 'b0': 250, 'sigma0': 20, 'C0': n*100}, {'A0': n*10000, 'b0': 820, 'sigma0': 50, 'C0': n*40},
                           {'A0': n*30000, 'b0': 950, 'sigma0': 40, 'C0': n*100}], 
                    'Co': {'A0': n*720000, 'b0': 350, 'sigma0': 20, 'C0': n*100, 'A1': n*5000, 'b1': 400, 'sigma1': 10}, 
                    'Cs': {'A0': n*12000, 'b0': 1700, 'sigma0': 60, 'C0': n*100}, 
                    'Na': {'A0': n*25000, 'b0': 1400, 'sigma0': 60, 'C0': n*100}}
sys_unc_channel = {'Ba': [(1 + 96.38-95.67)/2, 0.5 + (246.548-245.34)/2, 5 + (838.7-835.2)/2, 1 + (975.15-970.85)/2],'Co': [0.7 + (355.41-354.452)/2, 5 + (394.64-393.53)/2],'Cs': 2 + (1760.22-1752.09)/2,'Na': 1 + (1374.52-1368.61)/2}
el_peaks = {'75 ':[600, 790], '85 ':[650, 850], '95 ': [725, 925], '105':[800, 1025], '125':[975, 1300], '135':[1100, 1450], '220': [1250, 1500], '230': [1050, 1400], '240': [950, 1250], '250': [850,1100], '260': [750, 1000], '280': [640, 820]}
#0.4 for the other one in Barium
#__________________________________________________________________________________

def gaussian_func(x, A, b, sigma, C, B=0):
    ''' Function that approximates the shape of the peak (Gaussian) and its background (polynomial)
    Parameters:
    ----------
    x: int, channel at which we're evaluating the function
    A: int, number of counts in the peak
    b: int, center of the peak
    sigma: int, width of the peak
    C: int, constant background
    *Coeff: list, coefficient of polynomial backgrounds     
    Returns:
    --------
    value: number of counts due to the peak and the background 
    '''
    peak = A * np.exp(-1/2*((x - b)/sigma)**2)/(np.abs(sigma)*np.sqrt(2*np.pi))
    
    background = C + B*x
    
    return peak + background
#__________________________________________________________________________________

def two_gaussian_func(x, A, b, sigma, C, A1, b1, sigma1):
    ''' Function that approximates the shape of two peaks (Gaussian) and their background (constant)
    Parameters:
    ----------
    x: int, channel at which we're evaluating the function
    A: int, number of counts in the peak
    b: int, center of the peak
    sigma: int, width of the peak
    C: int, constant background 
    Returns:
    --------
    value: number of counts due to the peak and the background 
    '''
    peak1 = np.abs(A) * np.exp(-1/2*((x - b)/sigma)**2)/(np.abs(sigma)*np.sqrt(2*np.pi))
    peak2 = np.abs(A1) * np.exp(-1/2*((x - b1)/sigma1)**2)/(np.abs(sigma1)*np.sqrt(2*np.pi))
    
    background = C
    
    return peak1 + peak2 + background
#__________________________________________________________________________________

def three_gaussian_func(x, A, b, sigma, C, A1, b1, sigma1, A2, b2, sigma2):
    ''' Function that approximates the shape of two peaks (Gaussian) and their background (constant)
    Parameters:
    ----------
    x: int, channel at which we're evaluating the function
    A: int, number of counts in the peak
    b: int, center of the peak
    sigma: int, width of the peak
    C: int, constant background 
    Returns:
    --------
    value: number of counts due to the peak and the background 
    '''
    peak1 = np.abs(A) * np.exp(-1/2*((x - b)/sigma)**2)/(np.abs(sigma)*np.sqrt(2*np.pi))
    peak2 = np.abs(A1) * np.exp(-1/2*((x - b1)/sigma1)**2)/(np.abs(sigma1)*np.sqrt(2*np.pi))
    peak3 = np.abs(A2) * np.exp(-1/2*((x - b2)/sigma2)**2)/(np.abs(sigma2)*np.sqrt(2*np.pi))
    
    background = C
    
    return peak1 + peak2 + peak3 + background
#__________________________________________________________________________________

def gaussian_fit(data, region, A0=None, b0=None, sigma0=None, C0 = None, B0=0, 
                 A1 = None, b1 = None, sigma1 = None, A2 = None, b2 = None,
                 sigma2 = None, two_peaks = False, three_peaks = False, lin=False, unc = None):
    """ Function to fit Gaussian to compton scattering data. 
    
    Parameters:
    ----------
    data:               spinmob databox
    regions (optional): array of ints, domain of peak under consideration;
    A0 (optional):      int, initial guess for parameter A;
    b0 (optional):      int, initial guess for parameter b;
    sigma0 (optional):  int, initial guess for parameter sigma;
    C0 (optional): int, initial guess for background parameters 
    poly (optional): int, number of polynomial order
    multiple_peaks: True/False, whether to use multiple peaks for the fit
    
    Returns:
    --------
    param:              array of the gaussian fit parameters and uncertainties;  
    """
    f = s.data.fitter() # initiate fitter object
    if unc is None:
        unc = np.sqrt(np.abs(data['Counts']))
    
    if two_peaks:
        f.set_functions(f = two_gaussian_func, p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+', C='+str(C0)+', A1='+str(A1)+',b1='+str(b1)+',sigma1='+str(sigma1))
        
    elif three_peaks:
        f.set_functions(f = three_gaussian_func, p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+', C='+str(C0)
                        +', A1='+str(A1)+',b1='+str(b1)+',sigma1='+str(sigma1)+', A2='+str(A2)+',b2='+str(b2)+',sigma2='+str(sigma2))

    else:
        if lin:
            f.set_functions(f = gaussian_func, p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+', C='+str(C0)+', B='+str(B0))
        else:
            f.set_functions(f = gaussian_func, p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+', C='+str(C0))
        
    f.set_data(xdata = data['Channel'][region[0]:region[1]], ydata = data['Counts'][region[0]:region[1]], 
                  eydata = unc[region[0]:region[1]] + 1/10000, xlabel='Channel', ylabel='Counts')
    
    f.set(plot_guess=False)
    f.fit() # fit to data
    A = f.get_fit_results()['A']
    A_std = f.get_fit_results()['A.std']
    b = f.get_fit_results()['b']
    b_std = f.get_fit_results()['b.std']
    sigma = f.get_fit_results()['sigma']
    sigma_std = f.get_fit_results()['sigma.std']
    param = np.array([A, A_std, b, b_std, sigma, sigma_std])
    
    if two_peaks:
        b2 = f.get_fit_results()['b1']
        b2_std = f.get_fit_results()['b1.std']
        param = np.array([A, A_std, b, b_std, sigma, sigma_std, b2, b2_std])
    
    if three_peaks:
        b2 = f.get_fit_results()['b1']
        b2_std = f.get_fit_results()['b1.std']
        b3 = f.get_fit_results()['b2']
        b3_std = f.get_fit_results()['b2.std']
        param = np.array([A, A_std, b, b_std, sigma, sigma_std, b2, b2_std, b3, b3_std])
    
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
    f.set_data(xdata = energy, ydata = channel, eydata = channel_unc, xlabel='Energy (KeV)', ylabel='Channel') # supply the data
    f.set(plot_guess=False)
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
    databox['Counts'] -= databox['Counts'] # delete counts of saved databox
    for box in databoxes:
        databox['Counts'] += box['Counts'] # add all counts to saved databox

    return databox
#__________________________________________________________________________________

def calibrate(n, systematic=False):
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
            for j in range(1,3): # loop over all peaks of Barium
                _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, peaks[element][j], **gaussian_guesses[element][j]) # fit gaussian 
                if j == 0:
                    _, _, b, b_std, sigma, sigma_std, _, _ = gaussian_fit(data, peaks[element][j], **gaussian_guesses[element][j], 
                                                                          A1=n*25000, b1=94, sigma1=10, two_peaks=True) # fit two gaussians
                    energy = np.append(energy, energy_dict[element][j])
                    channel = np.append(channel, b)
                    if systematic: # taking into account systematic uncertainties 
                        channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element][j]) # width of peak
                    else: 
                        channel_unc = np.append(channel_unc, b_std ) # width of peak 
                    
                if j == 1:
                    _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, peaks[element][j], **gaussian_guesses[element][j], 
                                                                          B0 = -30, lin=True) # fit gaussian and linear background
                    energy = np.append(energy, energy_dict[element][j])
                    channel = np.append(channel, b)
                    if systematic:
                        channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element][j]) # width of peak
                    else:
                        channel_unc = np.append(channel_unc, b_std) # width of peak
                
                if j == 2:
                    _, _, b, b_std, sigma, sigma_std, b2, b2_std, _, _ = gaussian_fit(data, [peaks[element][2][0], peaks[element][3][1]], **gaussian_guesses['Ba'][3],
                                      A1=n*25000, b1=830, sigma1=30, A2=n*10000, b2 = 780, sigma2 = 30,
                                      three_peaks=True) # fit two gaussian gaussian 
                    energy = np.append(energy, energy_dict[element][2])
                    channel = np.append(channel, b2)
                    energy = np.append(energy, energy_dict[element][3])
                    channel = np.append(channel, b)
                    if systematic:
                        channel_unc = np.append(channel_unc, b2_std + sys_unc_channel[element][2])
                        channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element][3])
                    else: 
                        channel_unc = np.append(channel_unc, b2_std)
                        channel_unc = np.append(channel_unc, b_std)
        
        elif element == 'Co':
            _, _, b, b_std, sigma, sigma_std, b2, b2_std = gaussian_fit(data, peaks[element], **gaussian_guesses[element], two_peaks=True) # fit gaussian 
            energy = np.append(energy, energy_dict[element][0])
            channel = np.append(channel, b)
            energy = np.append(energy, energy_dict[element][1])
            channel = np.append(channel, b2)
            if systematic:
                channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element][0]) # width of peak 
                channel_unc = np.append(channel_unc, b2_std + sys_unc_channel[element][1]) # width of peak 
            else:
                channel_unc = np.append(channel_unc, b_std) # width of peak 
                channel_unc = np.append(channel_unc, b2_std) # width of peak 

        else:
            _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, peaks[element], **gaussian_guesses[element], B0=0, lin=True) # fit gaussian 
            energy = np.append(energy, energy_dict[element])
            channel = np.append(channel, b)
            if systematic:
                channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element]) # width of peak
            else: 
                channel_unc = np.append(channel_unc, b_std) # width of peak
        
    param = linear_fit(channel, channel_unc, energy) # do a linear fit for the channels computed against energy
    
    return param
#__________________________________________________________________________________

def energy_fit(energy, energy_unc, angle):
    """ Fit function for the energy again angles. Used in "get_rest_mass()" function.
    Arguments
    ---------
    energy:     array, energy peaks (obtained from the calibration values relating channel to energy);
    energy_unc: array, uncertainty in energy;
    angle:      array, angle under consideration;
    
    Return
    ------
    param:      fit parameters of energy vs angle;
    """

    f = s.data.fitter() # create a fitter object
    f.set_functions(f = '661.657/(1 + 661.657/E*(1-cos(pi*(x-x0)/180)))', p = 'E=511, x0 = 180') # set the function, with E = m_e c^2 (electron rest mass energy)
    f.set_data(xdata = angle, ydata = energy, eydata = energy_unc, xlabel='Angle (°)', ylabel='$Energy (KeV)$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit    
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std']] # fit parameters
     
    return param
#__________________________________________________________________________________

def energy_chan_fit(chan, chan_unc, angle):
    """ Fit function for the channel against angles.
    Arguments
    ---------
    channel:     array, channel peaks
    chan_unc: array, uncertainty in channel;
    angle:      array, angle under consideration;
    
    Return
    ------
    param:      fit parameters of energy vs angle;
    """

    f = s.data.fitter() # create a fitter object
    f.set_functions(f = 'A/(1 + A/E*(1-cos(pi*(x-x0)/180))) + B', p = 'A=1740, E=1000, x0 = 180, B = 50') # set the function, with E = m_e c^2 (electron rest mass energy)
    f.set_data(xdata = angle, ydata = chan, eydata = chan_unc, xlabel='Angle (°)', ylabel='$Channel$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit    
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std']] # fit parameters
     
    return param
#__________________________________________________________________________________

def inv_energy_fit(energy, energy_unc, angle):
    """ Fit function for the energy again angles. Used in "get_rest_mass()" function.
    Arguments
    ---------
    energy:     array, energy peaks (obtained from the calibration values relating channel to energy);
    energy_unc: array, uncertainty in energy;
    angle:      array, angle under consideration;
    
    Return
    ------
    param:      fit parameters of energy vs angle;
    """
    f = s.data.fitter() # create a fitter object
    f.set_functions(f = '1/661.6 + 1/E*(1-cos(pi*(x-x0)/180))', p = 'E=511, x0 = 180') # set the function, with E = m_e c^2 (electron rest mass energy)
    f.set_data(xdata = angle, ydata = energy**(-1), eydata = energy_unc * energy**(-2), xlabel='Angle (°)', ylabel='$Energy^{-1} (KeV^{-1})$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit    
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std']] # fit parameters
     
    return param
#__________________________________________________________________________________

def get_rest_mass(n, m, m_std, c, c_std, lin = True, chan_fit = False):
    """ Using the "energy_fit()" function above, fit the relation for energy against angle for angles selected. This function also requires background data to be selected 
    following the selection of a scatterer at a certain angle (background data must be at the same angle).
    Example for n=3 (fit for three different angles, say 220, 230, 240):
        1. launch the function,
        2. select files to combine for the first angle (say Al 220),
        3. select background data files to combine for first angle (no scatterer 220),
        4. select files to combine for the second angle (say Al 230),
        5. select background data files to combine for second angle (no scatterer 230),
        6. select files to combine for the third angle (say Al 240),
        7. select background data files to combine for third angle (no scatterer 240).
    Argument
    --------
    n:  int, number of different angles to consider;
    m:  float, slope of energy vs channel linear curve (computed through the calibration() function);
    c:  float, intercept of energy vs channel linear curve;
    lin: True for fitting against the energy and False for fitting against the inverse
    Return
    ------
    param: array, fit parameters from the energy_fit function;
    """
    print('Select all files associated to one scattered and angle, for multiple angles.')
    energy = np.zeros(n)
    energy_unc = np.zeros(n)
    chan = np.zeros(n)
    chan_unc = np.zeros(n)
    angles = []
    for i in range(0, n):
        data, unc = subtract_background()  # combine files of the same angle and scatterer
        angles.append(data.headers['description'][4:7]) # retrieve angle 
        element = data.headers['description'][0:2]
        if element == 'Al' or element == 'Cu':
            _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, el_peaks[angles[i]], 400, np.mean(el_peaks[angles[i]]), 30, 5, unc=unc)
        else: 
            print('Energy fit for' + element + ' not yet implemented.')
            return        
        chan[i] = b
        chan_unc[i] = b_std
        energy[i] = m*b+c 
        energy_unc[i] = np.sqrt(m**2*(b_std)**2) + 1/1000
        
        # Do we need to add the systematic uncertainty for cesium to b_std
        # to take into account the systematic uncertainties we had found for the calibration peak (shift and choice of background function)
        # + sys_unc_channel['Cs']
    angles = np.array([int(angle) for angle in angles])
    if chan_fit:
        param = energy_chan_fit(chan, chan_unc, angles)
    else:
        if lin:
            param = energy_fit(energy, energy_unc, angles)
        else:
            param = inv_energy_fit(energy, energy_unc, angles)

    return param
#__________________________________________________________________________________

def subtract_background(paths=None):
    """ Subtract background data of scattered data. Choose files which are of the same angle. """
    if paths is None:
        data = combine_chns()
        background = combine_chns()
    else:
        data = mcphysics.load
    unc = np.sqrt(data['Counts'] + background['Counts'])
    data['Counts'] -= background['Counts']

    return (data, unc)
#__________________________________________________________________________________

def get_sys_unc(n_regions):
    """ Estimate the systematic uncertainty in the location of our peaks by varying the 
    fits (there region and function we're fitting against) and computing the
          standard deviation of the resulting distribution.
        1. launch the function,
        2. select files to combine for the peak we want the systematic uncertainty of,
        3. returns the systematic uncertainty (if Ba was selected, returns one for each peaks)
    Argument
    --------
    n_regions:  int, number of different regions to consider
    Return
    ------
    sys_unc: number, estimate of the systematic uncertainty;
    """
    data = combine_chns()
    element = data.headers['description'][0:2]
    
    if element == 'Ba':
        distribution_b = np.zeros((n_regions,4))
        
        for i in range(n_regions):
            for j in range(4): # loop over all peaks of Barium
                _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, 
                        [peaks[element][j][0]-i*10, peaks[element][j][1]+i*10], **gaussian_guesses[element][j]) # fit gaussian 
                distribution_b[i, j] = b
                
        sys_unc = np.std(distribution_b, axis=0, ddof=1)
    
    elif element == 'Co':
        distribution_b = np.zeros((n_regions,2))
        for i in range(n_regions):
            _, _, b, b_std, sigma, sigma_std, b2, b2_std = gaussian_fit(data, [peaks[element][0]-i*5, peaks[element][1]+i*5],
                                                          **gaussian_guesses[element], two_peaks=True) # fit gaussian 
            distribution_b[i, 0] = b
            distribution_b[i, 1] = b2
            
        sys_unc = np.std(distribution_b, axis=0, ddof=1)
        
    else:
        distribution_b = np.zeros(n_regions)
        for i in range(n_regions):
            _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, [peaks[element][0]-i*10, peaks[element][1]+i*10],
                                                          **gaussian_guesses[element]) # fit gaussian 
            distribution_b[i] = b
            
        sys_unc = np.std(distribution_b, ddof=1)
            
    return sys_unc