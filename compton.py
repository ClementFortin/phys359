import spinmob as s
import mcphysics
import numpy as np

peaks = {'Ba': [[50, 130], [200, 300], [900, 1050]], 'Co': [335, 450], 'Na': [1250, 1600], 'Cs': [1625, 2000]}
energy_dict = {'Ba': [30.973, 80.9979, 356.0129], 'Co': [122.0605, 136.47356], 'Cs': 661.657, 'Na': 511.0}
gaussian_guesses = {'Ba': [{'A0': 100000, 'b0': 100, 'sigma0': 10, 'C0': 500}, {'A0': 100000, 'b0': 250, 'sigma0': 20, 'C0': 500}, {'A0': 100000, 'b0': 950, 'sigma0': 40, 'C0': 500}], 
                    'Co': {'A0': 700000, 'b0' : 360, 'sigma0' : 20, 'C0' : 0, 'd0' : 400, 'e0' : 10,'g0' : 5000}, 
                    'Cs': {'A0': 100000, 'b0': 1700, 'sigma0': 60, 'C0': 50}, 
                    'Na': {'A0': 100000, 'b0': 1400, 'sigma0': 50, 'C0': 50}}
al_peaks = {'220': [1250, 1500], '230': [1050, 1400], '240': [950, 1250], '250': [850,1100], '260': [750, 1000], '280': [650, 825]}
#__________________________________________________________________________________

def gaussian_fit(data, region, A0='', b0='', sigma0='', C0=''):
    """ Function to fit to photopeak in compton scattering data. 
    
    Arguments
    ---------
    data:               spinmob databox;
    regions:            array of ints, domain of peak under consideration;
    A0 (optional):      float, initial guess for parameter A;
    b0 (optional):      float, initial guess for parameter b;
    sigma0 (optional):  float, initial guess for parameter sigma;
    C0 (optional):      float, initial guess for parameter C;
    
    Returns:
    --------
    param:              array of the gaussian fit parameters and uncertainties;  
    """
    f = s.data.fitter() # initiate fitter object
    f.set_functions(f = 'A * exp(-0.5*((x - b)/sigma)**2)/(sigma*sqrt(2*pi)) + C', p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+', C='+str(C0))
    f.set_data(xdata = data['Channel'][region[0]:region[1]], ydata = data['Counts'][region[0]:region[1]]+0.01, eydata=np.sqrt(np.abs(data['Counts'][region[0]:region[1]])+0.01), xlabel='Channel', ylabel='Counts')
    f.set(plot_guess=False)
    f.fit() # fit to data
    param = np.array([f.get_fit_results()['A'], f.get_fit_results()['A.std'], f.get_fit_results()['b'], f.get_fit_results()['b.std'], f.get_fit_results()['sigma'], f.get_fit_results()['sigma.std'], f.get_fit_results()['C'], f.get_fit_results()['C.std']])
    
    return param
#__________________________________________________________________________________

def bimodal_fit(data, region, A0='', b0='', sigma0='', C0='', d0='', e0='', g0=''):
    """ Function to fit double photopeak in compton scattering data.

    Arguments
    ---------
    data:           spinmob databox;
    regions:            array of ints, domain of peak under consideration;
    A0 (optional):      float, initial guess for parameter A;
    b0 (optional):      float, initial guess for parameter b;
    sigma0 (optional):  float, initial guess for parameter sigma;
    C0 (optional):      float, initial guess for parameter C;
    d0 (optional):      float, initial guess for parameter d;
    e0 (optional):      float, initial guess for parameter e;

    Return
    ------
    param:              array of the bimodal fit parameters and uncertainties;

    """
    f = s.data.fitter()
    f.set_functions(f = 'A * exp(-0.5*((x - b)/sigma)**2)/(sigma*sqrt(2*pi))+ C* exp(-0.5*((x - d)/e)**2)/(e*sqrt(2*pi))+g', 
                    p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+',C='+str(C0)+',d='+str(d0)+',e='+str(e0)+', g='+str(g0))
    f.set_data(xdata = data['Channel'][region[0]:region[1]], ydata = data['Counts'][region[0]:region[1]]+0.01, eydata = np.sqrt(np.abs(data['Counts'][region[0]:region[1]])+0.01), xlabel='Channel', ylabel='Counts')
    f.set(plot_guess=False)
    f.fit()
    param = np.array([f.get_fit_results()['A'], f.get_fit_results()['A.std'], f.get_fit_results()['b'], f.get_fit_results()['b.std'], f.get_fit_results()['sigma'], f.get_fit_results()['sigma.std'], 
                    f.get_fit_results()['C'], f.get_fit_results()['C.std'], f.get_fit_results()['d'], f.get_fit_results()['d.std'], f.get_fit_results()['e'], f.get_fit_results()['e.std'], f.get_fit_results()['g'], f.get_fit_results()['g.std']])
    return param
#__________________________________________________________________________________
def trimodal_fit(data, region, A0='', b0='', sigma0='', C0='', d0='', e0='', G0='', h0='', j0='', k0=''):
    """ Function to fit double photopeak in compton scattering data.

    Arguments
    ---------
    data:           spinmob databox;
    regions:            array of ints, domain of peak under consideration;
    A0 (optional):      float, initial guess for parameter A;
    b0 (optional):      float, initial guess for parameter b;
    sigma0 (optional):  float, initial guess for parameter sigma;
    C0 (optional):      float, initial guess for parameter C;
    d0 (optional):      float, initial guess for parameter d;
    e0 (optional):      float, initial guess for parameter e;

    Return
    ------
    param:              array of the bimodal fit parameters and uncertainties;

    """
    f = s.data.fitter()
    f.set_functions(f = 'A * exp(-0.5*((x - b)/sigma)**2)/(sigma*sqrt(2*pi))+ C* exp(-0.5*((x - d)/e)**2)/(e*sqrt(2*pi))+G*exp(-0.5*((x - h)/j)**2)/(j*sqrt(2*pi))+k', 
                    p = 'A='+str(A0)+',b='+str(b0)+',sigma='+str(sigma0)+',C='+str(C0)+',d='+str(d0)+',e='+str(e0)+', G='+str(G0)+',h='+str(h0)+',j='+str(j0)+',k='+str(k0))
    f.set_data(xdata = data['Channel'][region[0]:region[1]], ydata = data['Counts'][region[0]:region[1]]+0.01, eydata = np.sqrt(np.abs(data['Counts'][region[0]:region[1]])+0.01), xlabel='Channel', ylabel='Counts')
    f.set(plot_guess=False)
    f.fit()
    param = np.array([f.get_fit_results()['A'], f.get_fit_results()['A.std'], f.get_fit_results()['b'], f.get_fit_results()['b.std'], f.get_fit_results()['sigma'], f.get_fit_results()['sigma.std'], 
                    f.get_fit_results()['C'], f.get_fit_results()['C.std'], f.get_fit_results()['d'], f.get_fit_results()['d.std'], f.get_fit_results()['e'], f.get_fit_results()['e.std'],
                    f.get_fit_results()['G'], f.get_fit_results()['G.std'], f.get_fit_results()['h'], f.get_fit_results()['h.std'], f.get_fit_results()['j'], f.get_fit_results()['j.std'], f.get_fit_results()['k'], f.get_fit_results()['k.std']])
    return param
#__________________________________________________________________________________

def inv_linear_fit(channel, channel_unc, energy):
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
                _, _, b, b_std, _, _, _, _ = gaussian_fit(data, peaks[element][j], **gaussian_guesses[element][j]) # fit gaussian 
                energy = np.append(energy, energy_dict[element][j])
                channel = np.append(channel, b)
                channel_unc = np.append(channel_unc, b_std) # width of peak
        elif element == 'Co':
            _, _, b, b_std, _, _, _, _, d, d_std, _, _, _, _ = bimodal_fit(data, peaks[element], **gaussian_guesses[element]) # fit bimodal gaussian
            energy = np.append(energy, energy_dict[element])
            channel = np.append(channel, [b,d]) # peaks
            channel_unc = np.append(channel_unc, [b_std, d_std]) # width of peaks
        else:
            _, _, b, b_std, _, _, _, _ = gaussian_fit(data, peaks[element], **gaussian_guesses[element]) # fit gaussian 
            energy = np.append(energy, energy_dict[element])
            channel = np.append(channel, b)
            channel_unc = np.append(channel_unc, b_std) # width of peak
        
    param = inv_linear_fit(channel, channel_unc, energy) # do a linear fit for the channels computed against energy
    
    return param
#__________________________________________________________________________________

def energy_fit(energy, energy_unc, angle, lin=True):
    """ Fit function for the energy again angles. Used in "get_rest_mass()" function.
    Arguments
    ---------
    energy:     array, energy peaks (obtained from the calibration values relating channel to energy);
    energy_unc: array, uncertainty in energy;
    angle:      array, angle under consideration;
    lin:        bool, fit E (True) or 1/E (False) vs angle;
    
    Return
    ------
    param:      fit parameters of energy vs angle;
    """
    f = s.data.fitter() # create a fitter object
    if lin:
        f.set_functions(f = '661.6/(1 + 661.6/E*(1-cos(x)))', p = 'E=511') # set the function, with E = m_e c^2 (electron rest mass energy)
        f.set_data(xdata = angle, ydata = energy, eydata = energy_unc, xlabel='Angle (°)', ylabel='$Energy (KeV)$') # supply the data
    else:
        f.set_functions(f = '1/661.6 + 1/E*(1-cos(x))', p = 'E=511') # set the function with 1/E, with E = m_e c^2 (electron rest mass energy)
        f.set_data(xdata = angle, ydata = energy**(-1), eydata = energy_unc * energy**(-2), xlabel='Angle (°)', ylabel='$Energy^{-1} (KeV^{-1})$') # supply the data
    
    f.set(plot_guess=False)
    f.fit() # make the fit
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std']] # fit parameters
     
    return param
#__________________________________________________________________________________

def get_rest_mass(n, m, m_std, c, c_std, lin = True):
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
    angles = []
    for i in range(0, n):
        data = subtract_background() # combine files of the same angle and scatterer
        angles.append(data.headers['description'][4:7]) # retrieve angle 
        element = data.headers['description'][0:2]
        if element == 'Al':
            _, _, b, b_std, sigma, sigma_std, _, _ = gaussian_fit(data, al_peaks[angles[i]], A0=400, b0=np.mean(al_peaks[angles[i]]), sigma0=30, C0=5)
        else: 
            print('Energy fit for' + element + ' not yet implemented.')
            return
        
        energy[i] = m*b+c 
        energy_unc[i] = np.sqrt(b**2*m_std**2 + m**2*b_std**2 + c_std**2)

    angles_int = np.array([int(angle) - 180 for angle in angles])
    param = energy_fit(energy, energy_unc, angles_int, lin)

    return param
#__________________________________________________________________________________

def subtract_background():
    """ Subtract background data of scattered data. Choose files which are of the same angle. """
    data = combine_chns()
    background = combine_chns()
    data['Counts'] -= background['Counts']

    return data

