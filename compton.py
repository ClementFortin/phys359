import spinmob as s
import mcphysics
import numpy as np
import matplotlib.pyplot as plt
from sympy import true

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
new_al_peaks = {'15 ': (373, 506), '25 ': (379, 522), '35 ': (412, 536), '45 ': (420, 576), '85 ': (584, 805), '125': (911, 1229), '135': (1036, 1354), '145': (1137, 1448), '190': (1038, 1370), '200': (957, 1262), '220': (748, 1033), '260': (495, 676), '300': (382, 526), '310': (371, 501), '320': (354, 485)}
cu_peaks = {'55 ': (510, 704), '65 ': (524, 765), '75 ': (567, 816), '85 ': (619, 877), '95 ': (687, 962), '105': (768, 1051), '125': (950, 1314), '135': (1080, 1465), '220': (1171, 1551), '230': (1009, 1430), '240': (918, 1287), '250': (827, 1136), '260': (755, 1016), '280': (619, 836), '300': (512, 744), '310': (469, 710)}
al_peaks = {'55 ': [525, 700], '65 ': [550, 750], '75 ':[600, 790], '85 ':[650, 850], '95 ': [725, 925], '105':[800, 1025], '125':[975, 1300], '135':[1100, 1450], '220': [1200, 1550], '230': [1050, 1400], '240': [950, 1250], '250': [850,1100], '260': [750, 1000], '280': [640, 820], '300': [500, 750], '310':[500, 775]}
chan_data = np.array([
    [   55.        ,   606.5848648 ,     0.66363029,    7790.,  330.], # [angle | peak channel | peak channel unc | total counts | total counts unc]
    [   65.        ,   645.46423482,     0.75686856,    7610.,  330.],
    [   75.        ,   689.84422013,     0.76523447,    8820.,  600.],
    [   85.        ,   747.45967702,     0.91398185,    6810.,  470.],
    [   95.        ,   823.9814263 ,     0.97691572,    6790.,  620.],
    [  105.        ,   909.71361641,     1.01943716,    6850.,  500.],
    [  125.        ,  1134.32276321,     0.68136989,    9790.,  350.],
    [  135.        ,  1271.81660525,     0.6732421 ,    11770., 430.],
    [  220.        ,  1359.97663422,     0.64041472,    15130., 470.],
    [  230.        ,  1220.62839242,     0.64798157,    11940., 370.],
    [  240.        ,  1101.44105453,     0.85161053,    10500., 560.],
    [  250.        ,   980.45616526,     0.83807919,    7890.,  470.],
    [  260.        ,   883.58265553,     0.85533126,    7100.,  370.],
    [  280.        ,   727.92472533,     0.73575519,    7200.,  660.],
    [  300.        ,   629.94660331,     0.7988351 ,    7330.,  240.],
    [  310.        ,   590.00164086,     0.72644449,    7130.,  380.]])

NaIEfficiency_data = np.array([
    [226.144466185702,  99.4117647058823],
    [255.280442425227,  98.4313725490196],
    [279.006316237724,  97.0588235294117],
    [300.04955185689,   95.4901960784313],
    [349.832093577503,  92.3529411764705],
    [382.345638402671,  90              ],
    [401.336604423706,  88.8235294117647],
    [442.195211258821,  86.2745098039215],
    [471.719913821329,  84.5098039215686],
    [503.215935925999,  82.9411764705882],
    [563.478296504828,  80              ],
    [601.100886440558,  78.4313725490196],
    [700.832180063009,  74.9019607843137],
    [804.013161116785,  71.5686274509803],
    [864.653502950036,  70              ],
    [900.297335695209,  69.0196078431372],
    [999.999999999999,  66.6666666666666]])
#0.4 for the other one in Barium
#__________________________________________________________________________________

def get_peak_data(chan=True, m=0.3824, c=-13.43):
    """Obtain peak data for each of the 16 angles, either in units of channels or in keV.
    
    Arguments
    ---------
    chan (opt):  bool, channel if True, kev if False;
    m    (opt):  float, slope from calibration;
    c    (opt):  float, intercept from calibration;
    
    Return
    ------
    data with [angles | energy | energy_unc]
    """    
    if chan:
        return chan_data
    else:
        kev_data = chan_data
        kev_data[:,1] = m*chan_data[:,1]+c 
        kev_data[:,2] = np.sqrt(m**2*(chan_data[:,2])**2)
        return kev_data

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

def linear_func(x, m, b):
    """Reciprocal of a linear function"""
    return x/m - b/m
#__________________________________________________________________________________

def compton_kev_func(x, x0, E):
    """"Energy compton function"""
    return 661.657/(1 + 661.657/E*(1-np.cos(np.pi*(x-x0)/180)))
#__________________________________________________________________________________

def compton_chan_func(x, x0, E):
    """"Energy compton function"""
    return 1756.1/(1 + 1756.1/E*(1-np.cos(np.pi*(x-x0)/180)))
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
        A1 = f.get_fit_results()['A1']
        A1_std = f.get_fit_results()['A1.std']
        b1 = f.get_fit_results()['b1']
        b1_std = f.get_fit_results()['b1.std']
        sigma1 = f.get_fit_results()['sigma1']
        sigma1_std = f.get_fit_results()['sigma1.std']
        C = f.get_fit_results()['C']
        C_std = f.get_fit_results()['C.std']
        param = np.array([A, A_std, b, b_std, sigma, sigma_std, A1, A1_std, b1, b1_std, sigma1, sigma1_std, C, C_std])
    
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
    f.set_functions(f = linear_func, p = 'm, b') # set the function as the reciprocal of a linear function
    f.set_data(xdata = energy, ydata = channel, eydata = channel_unc, xlabel='Energy (KeV)', ylabel='Channel') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit
    # Save fit parameters and uncertainties
    param = np.array([f.get_fit_results()['m'], f.get_fit_results()['m.std'], f.get_fit_results()['b'], f.get_fit_results()['b.std']])

    return param
#__________________________________________________________________________________

def combine_chns(time=False):
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
    if time:
        return (databox, len(databoxes))
    else: 
        return databox
#__________________________________________________________________________________

def calibrate(n, new=False, fit=True, systematic=False):
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
        if new:
            if element == 'Ba':
                for j in range(1,3): # loop over all peaks of Barium
                    _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, new_peaks[element][j], **new_gaussian_guesses[element][j]) # fit gaussian 
                    if j == 0:
                        _, _, b, b_std, sigma, sigma_std, _, _ = gaussian_fit(data, new_peaks[element][j], **new_gaussian_guesses[element][j], 
                                                                            A1=n*25000, b1=94, sigma1=10, two_peaks=True) # fit two gaussians
                        energy = np.append(energy, energy_dict[element][j])
                        channel = np.append(channel, b)
                        if systematic: # taking into account systematic uncertainties 
                            channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element][j]) # width of peak
                        else: 
                            channel_unc = np.append(channel_unc, b_std ) # width of peak 
                        
                    if j == 1:
                        _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, new_peaks[element][j], **new_gaussian_guesses[element][j], 
                                                                            B0 = -30, lin=True) # fit gaussian and linear background
                        energy = np.append(energy, energy_dict[element][j])
                        channel = np.append(channel, b)
                        if systematic:
                            channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element][j]) # width of peak
                        else:
                            channel_unc = np.append(channel_unc, b_std) # width of peak
                    
                    if j == 2:
                        _, _, b, b_std, sigma, sigma_std, b2, b2_std, _, _ = gaussian_fit(data, [new_peaks[element][2][0], new_peaks[element][3][1]], **new_gaussian_guesses['Ba'][3],
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
                _, _, b, b_std, sigma, sigma_std, _, _, b2, b2_std, _, _, _, _ = gaussian_fit(data, new_peaks[element], **new_gaussian_guesses[element], two_peaks=True) # fit gaussian 
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
                _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, new_peaks[element], **new_gaussian_guesses[element], B0=0, lin=True) # fit gaussian 
                energy = np.append(energy, energy_dict[element])
                channel = np.append(channel, b)
                if systematic:
                    channel_unc = np.append(channel_unc, b_std + sys_unc_channel[element]) # width of peak
                else: 
                    channel_unc = np.append(channel_unc, b_std) # width of peak
        else: # old calibration
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
                _, _, b, b_std, sigma, sigma_std, _, _, b2, b2_std, _, _, _, _ = gaussian_fit(data, peaks[element], **gaussian_guesses[element], two_peaks=True) # fit gaussian 
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

    if fit:    
        param = linear_fit(channel, channel_unc, energy) # do a linear fit for the channels computed against energy
        return param
    else:
        return channel, channel_unc, energy
#__________________________________________________________________________________
def fixed_energy_fit(energy, energy_unc, angle):
    """ Fit function for the energy against angles with angle offset fixed. Used in "get_rest_mass()" function.
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
    f.set_functions(f = 'A/(1 + A/E*(1-cos(pi*x/180)))', p = 'E=511, A=661.657') # set the function, with E = m_e c^2 (electron rest mass energy)
    f.set_data(xdata = angle, ydata = energy, eydata = energy_unc, xlabel='Angle (°)', ylabel='$Energy (KeV)$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit    
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std'], f.get_fit_results()['A'], f.get_fit_results()['A.std']] # fit parameters
     
    return param

#__________________________________________________________________________________

def energy_fit(energy, energy_unc, angle):
    """ Fit function for the energy against angles. Used in "get_rest_mass()" function.
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
    f.set_data(xdata = angle, ydata = energy, eydata = energy_unc, xlabel='Angle (°)', ylabel='$Energy (keV)$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std'], f.get_fit_results()['x0'], f.get_fit_results()['x0.std']] # fit parameters
     
    return param
#__________________________________________________________________________________

def energy_chan_fit(chan, chan_unc, angle):
    """ Fit function for the channel against angles.
    Arguments
    ---------
    channel:    array, channel peaks
    chan_unc:   array, uncertainty in channel;
    angle:      array, angle under consideration;
    
    Return
    ------
    param:      fit parameters of energy vs angle;
    """

    f = s.data.fitter() # create a fitter object
    f.set_functions(f = '(1756.1+b)/(1 + (1756.1+b)/E*(1-cos(pi*(x-x0)/180)))', p = 'E=1300, x0 = 180') # set the function, with E = m_e c^2 (electron rest mass energy)
    f.set_data(xdata = angle, ydata = chan, eydata = chan_unc, xlabel='Angle (°)', ylabel='$Channel$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit    
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std'], f.get_fit_results()['x0'], f.get_fit_results()['x0.std']] # fit parameters
     
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
    f.set_data(xdata = angle, ydata = energy**(-1), eydata = energy_unc * energy**(-2), xlabel='Angle (°)', ylabel='$Energy^{-1} (keV^{-1})$') # supply the data
    f.set(plot_guess=False)
    f.fit() # make the fit    
    param = [f.get_fit_results()['E'], f.get_fit_results()['E.std']] # fit parameters
     
    return param
#__________________________________________________________________________________

def get_rest_mass(n1, n2=0, m1=0.3824, m1_std=0.001, c1=-13.43, c1_std=0.48, m2=0.4802, m2_std=0.0014, c2=-8.74, c2_std=0.49, lin =True, chan_fit=False, fit=True):
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
    energy = np.zeros(n1+n2)
    energy_unc = np.zeros(n1+n2)
    chan = np.zeros(n1+n2)
    chan_unc = np.zeros(n1+n2)
    angles = []
    for i in range(0, n1):
        data, unc = subtract_background()  # combine files of the same angle and scatterer
        angles.append(data.headers['description'][4:7]) # retrieve angle 
        element = data.headers['description'][0:2]
        if element == 'Al':
            _, _, b, b_std, _, _ = gaussian_fit(data, al_peaks[angles[i]], 40000, np.mean(al_peaks[angles[i]]), 30, 5, unc=unc, lin=True)
        elif element == 'Cu':
            _, _, b, b_std, _, _ = gaussian_fit(data, al_peaks[angles[i]], 40000, np.mean(cu_peaks[angles[i]]), 30, 5, unc=unc, lin=True)
        else: 
            print('Energy fit for' + element + ' not yet implemented.')
            return
        chan[i] = b
        chan_unc[i] = b_std
        energy[i] = m1*b+c1
        energy_unc[i] = np.sqrt(m1**2*(b_std)**2)

    subangles1 = np.array([int(angle) for angle in angles[0:n1]], dtype=np.float64)
    param1 = energy_fit(energy[0:n1], energy_unc[0:n1], subangles1)
    plt.close()
    subangles1 -= param1[2]
    if n2 > 0: 
        for j in range(0, n2):
            data, unc = subtract_background()  # combine files of the same angle and scatterer
            angles.append(data.headers['description'][4:7]) # retrieve angle 
            element = data.headers['description'][0:2]
            if element == 'Al':
                _, _, b, b_std, _, _ = gaussian_fit(data, new_al_peaks[angles[n1+j]], 40000, np.mean(new_al_peaks[angles[n1+j]]), 30, 5, unc=unc, lin=True)
            #elif element == 'Cu':
            #    _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, new_al_peaks[angles[i]], 40000, np.mean(new_al_peaks[angles[i]]), 30, 5, unc=unc, lin=True)
            else: 
                print('Energy fit for' + element + ' not yet implemented.')
                return
            chan[n1+j] = b
            chan_unc[n1+j] = b_std
            energy[n1+j] = m2*b+c2
            energy_unc[n1+j] = np.sqrt(m2**2*(b_std)**2)
        subangles2 = np.array([int(angle) for angle in angles[n1:n1+n2]], dtype=np.float64)
        param2 = energy_fit(energy[n1:n1+n2], energy_unc[n1:n1+n2], subangles2)
        plt.close()
        subangles2 -= param2[2]
        angles = np.hstack([subangles1, subangles2])
    else:
        angles = subangles1

    if fit: # plot energy
        if chan_fit:
            param = energy_chan_fit(chan, chan_unc, angles)
        else:
            if lin:
                param = fixed_energy_fit(energy, energy_unc, angles)
            else:
                param = inv_energy_fit(energy, energy_unc, angles)
        return param
    else:
        if chan_fit:
            return angles, chan, chan_unc
        else:
            return angles, energy, energy_unc
#__________________________________________________________________________________

def subtract_background(time = False):
    """ Subtract background data of scattered data. Choose files which are of the same angle. """
    data, Nfiles = combine_chns(time = True)
    background = combine_chns()
    unc = np.sqrt(data['Counts'] + background['Counts'])
    data['Counts'] -= background['Counts']

    if time:
        return [data, unc, Nfiles]
    else:
        return [data, unc]

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
#__________________________________________________________________________________

def plot_calibration(n, systematic=True):
    """Plot calibration using specified format."""
    channel, channel_unc, energy = calibrate(n, systematic=systematic, fit=False)
    param = linear_fit(channel, channel_unc, energy) # do a linear fit for the channels computed against energy
    plt.close() # close automatically generated plot from linear_fit() function
    # Making the plot
    extended_energy = np.linspace(min(energy)-50,max(energy)+50)
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.errorbar(energy, channel, yerr=channel_unc, fmt="o", label="Data")
    plt.plot(extended_energy, linear_func(extended_energy, param[0], param[2]), color='red', linestyle="-", label='Linear fit') # best estimate for linear fit
    plt.ylabel('Channel', Fontsize=18)
    plt.xlim(50, 700)
    plt.legend(loc="lower right", fontsize=16)
    plt.yticks(fontsize=14)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.grid()
    #Residual plot
    frame2=fig1.add_axes((.1,.1,.8,.2)) 
    plt.errorbar(energy, (channel-linear_func(energy, param[0], param[2]))/channel_unc, yerr=channel_unc/channel_unc, fmt='o', color='blue', alpha=0.75, zorder=0)
    plt.xlabel('Energy (keV)', Fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Studentized \n Residuals', Fontsize=18, labelpad=0)
    plt.xlim(50, 699)
    plt.ylim(-4,4)
    plt.yticks([-2,0,2])
    plt.hlines(0, 50, 700, color='red', linewidth=1, zorder=5)
    plt.grid()
#__________________________________________________________________________________
def plot_energy_angle(n, m, m_std, c, c_std, chan_fit=False):
    """Plot energy against angle to obtain a value for the rest mass energy"""
    
    angles, ydata, ydata_unc = get_rest_mass(n, m, m_std, c, c_std, lin=True, chan_fit=chan_fit, fit=False) # retrieve angles, energy and unc from rest mass function
    # obtain fit parameters
    if chan_fit:
        param = energy_chan_fit(ydata, ydata_unc, angles)
    else:
        param = energy_fit(ydata, ydata_unc, angles)
    plt.close() # close plot from fit
    extended_angles = np.linspace(min(angles)-10, max(angles)+10)
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.errorbar(angles-param[2], y=ydata, yerr=ydata_unc, fmt="o", label="Data")
    if chan_fit:
        plt.plot(extended_angles-param[2], compton_chan_func(extended_angles, param[2], param[0]), color='red', linestyle="-", label='Compton law fit') # best estimate for linear fit
        plt.ylabel('Channel', Fontsize=18)
    else:
        plt.errorbar(angles-param[2], (ydata-compton_kev_func(angles, param[2], param[0]))/ydata_unc, yerr=ydata_unc/ydata_unc, fmt='o', color='blue', alpha=0.75, zorder=0)
        plt.ylabel('Energy (keV)', Fontsize=18)
    plt.xlim(extended_angles[0]-param[2], extended_angles[-1]-param[2])
    plt.legend(loc="upper right", fontsize=16)
    plt.yticks(fontsize=14)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.grid()
    #Residual plot
    frame2=fig1.add_axes((.1,.1,.8,.2))
    if chan_fit:
        plt.errorbar(angles-param[2], (ydata-compton_chan_func(angles, param[2], param[0]))/ydata_unc, yerr=ydata_unc/ydata_unc, fmt='o', color='blue', alpha=0.75, zorder=0)
    else:
        plt.errorbar(angles-param[2], (ydata-compton_kev_func(angles, param[2], param[0]))/ydata_unc, yerr=ydata_unc/ydata_unc, fmt='o', color='blue', alpha=0.75, zorder=0)
    plt.xlabel('Angles (°)', Fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Studentized \n Residuals', Fontsize=18, labelpad=0)
    plt.xlim(extended_angles[0]-param[2], extended_angles[-1]-param[2])
    #plt.ylim(-4,4)
    plt.yticks([-20,-10,0,10])
    plt.ylim(-20,20)
    plt.hlines(0, extended_angles[0]-param[2], extended_angles[-1]-param[2], color='red', linewidth=1, zorder=5)
    plt.grid()
#__________________________________________________________________________________

def eff_corr(counts, ene):
    #eff = np.interp(ene, NaIEfficiency_data[:, 0], NaIEfficiency_data[:, 1])
    eff = 2.5E-12*ene**5-6.3E-09*ene**4+5.9E-06*ene**3-0.0023*ene**2+0.17*ene+95
    # Attenuation of air is negligible and photopeak to total ratio is not important

    return 100 * counts/eff
#__________________________________________________________________________________

def ThomFunc_fixed(x, N):
    #theta0 = 180.77 # offset in degrees
    I = 3.7 * 10**9 # initial activity of source in Bq (October 17 1974)
    t = 47.34 # time in years since ^^
    half_life = 30.05 # in years
    mean_life_factor = 1.4427
    p0 = 52.64 # cm
    num_photon_per_sec = 0.85
    r0 = 2.818E-13
    target_detector = 0.0175
    counts = N * (I * np.exp(-t/(half_life*mean_life_factor)) / (4 * np.pi * p0**2)) * num_photon_per_sec * 1/2 * (r0)**2 * (1+np.cos(np.pi*(x)/180)**2) * target_detector
    return counts

def ThomFunc(x, N, x0):
    #theta0 = 180.77 # offset in degrees
    I = 3.7 * 10**9 # initial activity of source in Bq (October 17 1974)
    t = 47.34 # time in years since ^^
    half_life = 30.05 # in years
    mean_life_factor = 1.4427
    p0 = 52.64 # cm
    num_photon_per_sec = 0.85
    r0 = 2.818E-13
    target_detector = 0.0175
    counts = N * (I * np.exp(-t/(half_life*mean_life_factor)) / (4 * np.pi * p0**2)) * num_photon_per_sec * 1/2 * (r0)**2 * (1+np.cos(np.pi*(x-x0)/180)**2) * target_detector
    return counts
    
#__________________________________________________________________________________


def ThomFit(counts, counts_unc, angle, fixed=False):
    """ Fit function for the Thomson cross section
    Arguments
    ---------
    counts:     array, counts peaks
    counts_unc: array, uncertainty in counts;
    angle:      array, angle under consideration;
    
    Return
    ------
    param:      Number of electrons / seconds;
    """

    f = s.data.fitter() # create a fitter object
    if fixed:
        f.set_functions(f = ThomFunc_fixed, p = 'N ='+str(10E25)) 
        f.set_data(xdata = angle, ydata = counts, eydata = counts_unc, xlabel='Scattered Angle (°)', ylabel='Counts') # supply the data
        f.set(plot_guess=False)
        f.fit() # make the fit    
        param = [f.get_fit_results()['N'], f.get_fit_results()['N.std']] # fit parameters
    else:
        f.set_functions(f = ThomFunc, p = 'N ='+str(10E25)+', x0 = 180') 
        f.set_data(xdata = angle, ydata = counts, eydata = counts_unc, xlabel='Scattered Angle (°)', ylabel='Counts') # supply the data
        f.set(plot_guess=False)
        f.fit() # make the fit    
        param = [f.get_fit_results()['N'], f.get_fit_results()['N.std'], f.get_fit_results()['x0'], f.get_fit_results()['x0.std']] # fit parameters
     
    return param

#__________________________________________________________________________________


def NKfunc_fixed(x, N, alpha):
    # no angle offset considered
    #alpha = 1.295
    #a = alpha * (1-np.cos(np.pi*(x)/180))
    return ThomFunc_fixed(x, N)/(1+alpha * (1-np.cos(np.pi*(x)/180)))**2*(1 + (alpha * (1-np.cos(np.pi*(x)/180)))**2/((1 + (np.cos(np.pi*(x)/180))**2) * (1 + alpha * (1-np.cos(np.pi*(x)/180)))))
    

def NKfunc(x, N, alpha, x0):
    # no angle offset considered
    #alpha = 1.295
    #a = alpha * (1-np.cos(np.pi*(x-x0)/180))
    return ThomFunc(x, N, x0)/(1+alpha * (1-np.cos(np.pi*(x-x0)/180)))**2*(1 + (alpha * (1-np.cos(np.pi*(x-x0)/180)))**2/((1 + (np.cos(np.pi*(x-x0)/180))**2) * (1 + alpha * (1-np.cos(np.pi*(x-x0)/180)))))
  
#__________________________________________________________________________________


def NKfit(counts, counts_unc, angle, fixed=False):
    """ Fit function for the NK cross section
    Arguments
    ---------
    counts:     array, counts peaks
    counts_unc: array, uncertainty in counts;
    angle:      array, angle under consideration;
    
    Return
    ------
    param:      Number of electrons / seconds;
    """
    f = s.data.fitter() # create a fitter object
    if fixed:
        f.set_functions(f = NKfunc_fixed, p = 'N ='+str(10E25)+', alpha = 1.3') 
        f.set_data(xdata = angle, ydata = counts, eydata = counts_unc, xlabel='Scattered Angle (°)', ylabel='Counts') # supply the data
        f.set(plot_guess=False)
        f.fit() # make the fit
        param = [f.get_fit_results()['N'], f.get_fit_results()['N.std'], f.get_fit_results()['alpha'], f.get_fit_results()['alpha.std']] # fit parameters        
    else:
        f.set_functions(f = NKfunc, p = 'N ='+str(10E25)+', alpha = 1.3, x0 = 180') 
        f.set_data(xdata = angle, ydata = counts, eydata = counts_unc, xlabel='Scattered Angle (°)', ylabel='Counts') # supply the data
        f.set(plot_guess=False)
        f.fit() # make the fit
        param = [f.get_fit_results()['N'], f.get_fit_results()['N.std'], f.get_fit_results()['alpha'], f.get_fit_results()['alpha.std'], f.get_fit_results()['x0'], f.get_fit_results()['x0.std']] # fit parameters
    
    return param

#__________________________________________________________________________________

def cross_fit(n1, n2=0, m1=0.3824, m1_std=0.001, c1=-13.43, c1_std=0.48, m2=0.4802, m2_std=0.0014, c2=-8.74, c2_std=0.49, Thom=False, fit=True):
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
    Return
    ------
    param: array, fit parameters from the thom function;
    """
    print('Select all files associated to one scattered and angle, for multiple angles.')
    intensity = np.zeros(n1+n2)
    intensity_unc = np.zeros(n1+n2)
    angles = []
    for i in range(0, n1):
        data, unc, Nfiles = subtract_background(time=True)  # combine files of the same angle and scatterer
        angles.append(data.headers['description'][4:7]) # retrieve angle 
        element = data.headers['description'][0:2]
        if element == 'Al':
            A, A_std, b, _, _, _ = gaussian_fit(data, al_peaks[angles[i]], 400, np.mean(al_peaks[angles[i]]), 30, 5, unc=unc, lin=True)
        elif element == 'Cu':
            A, A_std, b, _, _, _ = gaussian_fit(data, cu_peaks[angles[i]], 400, np.mean(cu_peaks[angles[i]]), 30, 5, unc=unc, lin=True)
        else: 
            print('Energy fit for' + element + ' not yet implemented.')
            return
        
        ene = m1*b+c1 
        intensity[i] = eff_corr(A, ene) / (30 * Nfiles)
        intensity_unc[i] = eff_corr(A_std, ene) / (30 * Nfiles)

    subangles1 = np.array([int(angle) for angle in angles[0:n1]], dtype=np.float64)
    if Thom:
        param1 = ThomFit(intensity[0:n1], intensity_unc[0:n1], subangles1, fixed=False)
    else:
        param1 = NKfit(intensity[0:n1], intensity_unc[0:n1], subangles1, fixed=False) 
    plt.close()
    subangles1 -= param1[4]

    if n2 > 0:
        for j in range(0, n2):
            data, unc, Nfiles = subtract_background(time=True)  # combine files of the same angle and scatterer
            angles.append(data.headers['description'][4:7]) # retrieve angle 
            element = data.headers['description'][0:2]
            if element == 'Al':
                A, A_std, b, _, _, _ = gaussian_fit(data, new_al_peaks[angles[j+n1]], 40000, np.mean(new_al_peaks[angles[j+n1]]), 30, 5, unc=unc, lin=True)
            elif element == 'Cu':
                A, A_std, b, _, _, _ = gaussian_fit(data, cu_peaks[angles[j+n1]], 40000, np.mean(cu_peaks[angles[j+n1]]), 30, 5, unc=unc, lin=True)
            else: 
                print('Energy fit for' + element + ' not yet implemented.')
                return
            
            ene = m2*b+c2 
            intensity[n1+j] = eff_corr(A, ene) / (30 * Nfiles)
            intensity_unc[n1+j] = eff_corr(A_std, ene) / (30 * Nfiles)

        subangles2 = np.array([int(angle) for angle in angles[n1:n1+n2]], dtype=np.float64)
        if Thom:
            param2 = ThomFit(intensity[n1:n1+n2], intensity_unc[n1:n1+n2], subangles2, fixed=False)
        else:
            param2 = NKfit(intensity[n1:n1+n2], intensity_unc[n1:n1+n2], subangles2, fixed=False)
        plt.close()
        subangles2 -= param2[4]
        angles = np.hstack([subangles1, subangles2])
    
    else:     
        angles = subangles1
    if fit:
        if Thom:
            param = ThomFit(intensity, intensity_unc, angles, fixed=True)
        else:
            param = NKfit(intensity, intensity_unc, angles, fixed=True)
        return param
    else:
        return angles, intensity, intensity_unc
#__________________________________________________________________________________

def get_region_bounds(n, k, lin = True, chan_fit = False):
    """ Determines the integration bounds of n angles by setting them as (b - k*sigma, b + k*sigma)
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
    k: float, number of stanfard deviation away from the average to include
    Return
    ------
    regions: dictionary, regions for each angle to integrate in;
    """
    print('Select all files associated to one scattered and angle, for multiple angles.')

    regions = {}
    angles = []
    for i in range(0, n):
        data, unc = subtract_background()  # combine files of the same angle and scatterer
        angles.append(data.headers['description'][4:7]) # retrieve angle 
        element = data.headers['description'][0:2]
        if element == 'Al':
            _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, al_peaks[angles[i]], 400, np.mean(al_peaks[angles[i]]), 30, 5, unc=unc)
        elif element == 'Cu':
            _, _, b, b_std, sigma, sigma_std = gaussian_fit(data, cu_peaks[angles[i]], 400, np.mean(cu_peaks[angles[i]]), 30, 5, unc=unc)
        else: 
            print('Energy fit for' + element + ' not yet implemented.')
            return
        
        regions.update({angles[i] : (int(b - k * sigma), int(b + k * sigma))})

    return regions

#__________________________________________________________________________________

def plot_cross_section():
    n = 16
    Thom = False
    N_al = 9.6E24 
    N_cu = 3E25
    time_sec = 150
    factor = 2

    extended_angles = np.linspace(min(angles)-10, max(angles)+10)
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.errorbar(angles-180.77, y=intensity, yerr=intensity_unc, fmt="o", label="Data")
    plt.plot(extended_angles-180.77, compton.ThomFunc(extended_angles, N_al/factor), color='red', linestyle="-", label='Thomson') # best estimate for linear fit
    plt.plot(extended_angles-180.77, compton.NKfunc(extended_angles, N_al/factor, 1.29), color='Green', linestyle="-", label='N-K') # best estimate for linear fit
    plt.ylabel('Detection Rate', Fontsize=18)
    plt.xlim(extended_angles[0]-180.77, extended_angles[-1]-180.77)
    plt.legend(loc="upper right", fontsize=16)
    plt.yticks(fontsize=14)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.grid()
    plt.yscale('linear')
    #Residual plot
    frame2=fig1.add_axes((.1,.1,.8,.2))
    if Thom:
        plt.errorbar(angles-180.77, (intensity/time_sec-compton.ThomFunc(angles, N_al/factor))/intensity_unc, yerr=intensity_unc/intensity_unc, fmt='o', color='blue', alpha=0.75, zorder=0)
    else:
        plt.errorbar(angles-180.77, time_sec*(intensity/time_sec-compton.NKfunc(angles, N_al/factor, 1.29))/intensity_unc, yerr=intensity_unc/intensity_unc, fmt='o', color='blue', alpha=0.75, zorder=0)
    plt.xlabel('Angles (°)', Fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Studentized \n Residuals', Fontsize=18, labelpad=0)
    plt.xlim(extended_angles[0]-180.77, extended_angles[-1]-180.77)
    #plt.ylim(-4,4)
    #plt.yticks([-20,-10,0,10])
    #plt.ylim(-20,20)
    plt.hlines(0, extended_angles[0]-180.77, extended_angles[-1]-180.77, color='red', linewidth=1, zorder=5)
    plt.grid()