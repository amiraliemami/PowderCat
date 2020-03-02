import numpy as np


###### smoothing functions ###############################################

def gauss_point(x, tau):
    # define gaussian memory function
    return (1/np.sqrt(2*np.pi*tau**2)) * np.exp(-(1/2)*(x/tau)**2)


def gaussian(x, y, tau):
    '''
    Function that returns the weighted average at each point x.

    IN: x     array of wavelengths
        y     array of intensities
        tau   parameter for gauss

    OUT: smoothed ys
    '''

    # subfunction that finds weighted avg at each point
    def weighted_avg(x0, y, tau):
        weight = gauss_point(x0-x, tau)
        # multiply each point by its weight and sum and divide to get avg at that point
        weighted_avg_point = np.sum(y*weight)/np.sum(weight)
        return weighted_avg_point

    return [weighted_avg(xi, y, tau) for xi in x]


def boxcar(y, p):

    n = len(y)
    y_new = []

    for i in range(0, n):

        lo = i - p
        hi = i + p

        # case for edges (only average available data points)
        if lo < 0:
            lo = 0
        if hi > n:
            hi = n

        # pick out window and its length
        window = y[lo:hi+1]
        w_len = len(window)
        y_new.append(sum(window)/w_len)

    return y_new


###### loading functions ##########################################


def load_radar(path):
    """
    Loads starcat style data.

    Parameters:
    path    abolute or relative path to file

    Returns:
    array of channel intensities in the file.
    """
    data = np.loadtxt(path)
    return data

def load_spec(path, ctype='multi', n=1):
    """
    Loads SpectraSuite style data.

    Parameters:
    path    abolute or relative path to file
    ctype   capture type, 'multi' (high speed) or 'single' (text delimited, NO header), default 'multi'

    Returns:
    list of frames (which are lists of intensities) in the file.
    ~ if 'single', then single frame is returned.
    """

    if ctype == 'multi':
        # skip first row because it's nonsense
        # skip second row because it is a row of zeros
        # then transpose to get first row be wavelengths and then each row being a reading
        # get rid of the wavelengths as they are not needed (always the same)
        if n > 1:
            data = np.loadtxt(path, skiprows=2).transpose()[1:]
        else:
            data = np.loadtxt(path, skiprows=2).transpose()[1]

    if ctype == 'single':
        # skip first row as this is the zero value
        # transpose and extract only the intensities, not the wavelengths
        data = np.loadtxt(path, skiprows=1).transpose()[1]

    return data


###### Processing ##########################################

def dark_sub(x, d):
    # subtract dark
    d_sub = x - d
    # make minimum 0.0001 to prevent div by zero
    d_sub = d_sub - min(d_sub) + 0.0000001
    return d_sub

def normalise(x):
    # make minimum 1
    x_new = x - min(x) + 0.0000001
    # normalise (gets rid of absolute magnitude information)
    normed = x_new/max(x_new)
    return normed


# all together - old
def process(raw, dark = None, norm = None, smooth = None, standard = None, derivative = None):
    """
    processes spectrum, return desired processed versions.
    NOTE: Defaults are None. If parameter not specified, it will not be produced.

    dark       numpyarray, dark frame. 
    norm       True or empty. (Dark frame must be supplied)
    smooth     int, number of pixels to each side for boxcar.
    standard   numpyarray, pre- dark-subbed and normed standard.
    derivative int, 1 or 2, returns 1st or 2nd derivative of the spectrum (TBI).
    """
        
    # make into numpy array, in case it ain't
    raw = np.array(raw)
    rv = []

    if dark is not None:
        # subtract dark
        d_sub = raw - dark
        # make minimum 1
        d_sub = d_sub - min(d_sub) + 1
        rv.append(d_sub)

        if norm is not None:
            # normalise (gets rid of absolute magnitude information)
            normed = d_sub/max(d_sub)
            rv.append(normed)

    if smooth is not None:
        p = smooth
        # boxcar smooth, p pixels each side
        smoothed = boxcar(raw, p)
        rv.append(smoothed)
    # gaussian smooth, tau = 1
    #g_smoothed = gaussian(w, raw, 1)

    if standard is not None:
        # calibrated absorption, divide by standard
        calib = normed/standard
        rv.append(calib)

    # derivative?
    
    ### IMPLEMENT DERIVATIVE

    # return desired processed frames
    return rv
