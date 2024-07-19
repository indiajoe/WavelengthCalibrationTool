#!/usr/bin/env python
""" This is a non-interactive tool to re-calibrate wavelength solution based on
an existing calibrated solution of the same lamp in same instrument."""
import sys
import argparse
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as optimize
import logging
from scipy.constants import speed_of_light
from .utils import calculate_cov_matrix_fromscipylsq
try:
    from skimage import registration
except ImportError:
    logging.warning('Failed to import scikit-image module for fast phase_cross_correlation. Fast phase_cross_correlation function will not work without that.')

try:
    from functools32 import partial
except ModuleNotFoundError:
    from functools import partial


def scale_interval_m1top1(x,a,b,inverse_scale=False):
    """ Scales input x in interval a to b to the range -1 to 1 """
    if inverse_scale: # convert form -1 to 1 scale back to a to b scale
        return ((b-a)*x + a+b)/2.0
    else:
        return (2.0*x - (b+a))/(b-a)

def LCTransformMatrixP(p,deg=6):
    """ Returns the Legendre coefficent transform matrix of pixel shift"""
    TM_pix = np.matrix(np.identity(deg+1))
    for i in range(TM_pix.shape[1]):
        for j in range(i+1,TM_pix.shape[0],2):
            TM_pix[i,j]=p*(i*2 +1)
    return TM_pix

def TransformLegendreCoeffs(LC,PVWdic,normP=False):
    """ Returns the transfromed Legendre coefficent based on the pixel shift, velocity shft, and Wavelength shift values 
    LC: Input coefficents
    PVWdic: Dictionary containing the transformation coefficents for p,v, and w.
    normP : If True will remove the scaling factor from pixel shift which is degenerate with velocity
    Returns :
    T_LC: Transformed LC = V*P*(LC+w)
    """
    ldeg = len(LC)-1
    w = np.zeros(ldeg+1)
    w[0] = PVWdic['w']
    T_LC = LC + w   # Wavl shift

    p = PVWdic['p']
    TM_pix = LCTransformMatrixP(p,deg=ldeg)
    Original_norm = np.linalg.norm(T_LC)
    T_LC = np.array(np.matmul(TM_pix,T_LC))[0]  # Pixel shift
    if normP:
        # Rescale to previous norm to break degeneracy with velocity shift
        T_LC /= np.linalg.norm(T_LC)/Original_norm 

    v = PVWdic['v']
    T_LC = T_LC * (1+v)  # Velocity shift

    return T_LC

def update_coeffs_with_defaults(coeffs,defaultParamDic=None):
    """ Returns the updated coeffs list with the values in defaultParamDic """
    if defaultParamDic:
        if isinstance(coeffs,tuple): # Change it to a list so that we can update
            coeffs = list(coeffs)
        for key in sorted(defaultParamDic.keys()):
            try:
                coeffs[key] = defaultParamDic[key]
            except IndexError:
                if len(coeffs) == key:
                    try:
                        coeffs.append(defaultParamDic[key])
                    except AttributeError: # if coeffs is a numpy array or tuple
                        coeffs = np.concatenate([coeffs, [defaultParamDic[key]]])
                else:
                    print('Missing number of coefficents ({0}) to add/insert default coeff {1} from {2}'.format(len(coeffs),key,defaultParamDic))
                    raise
    return coeffs
    
def transformed_spectrum(FluxSpec, *params, **kwargs):
    """ Returns transformed and interpolated scaled FluxSpec for fitting with data .
    Allowed **kwargs are 
           method: 'p' for normal polynomial
                   'c' for chebyshev  (default)
           WavlCoords : Coordinates to fit the transformation; (default: pixel coords)
    """
    
    if 'WavlCoords' in kwargs:
        Xoriginal = kwargs['WavlCoords']
    else:
        Xoriginal = np.arange(len(FluxSpec))

    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = 'c'

    if 'defaultParamDic' in kwargs:
        defaultParamDic = kwargs['defaultParamDic']
    else:
        defaultParamDic = None

    # First paramete is the flux scaling
    scaledFlux = FluxSpec*params[0]

    # Remaing parameters for defining the ploynomial drift of coordinates
    if len(params[1:]) == 1:  # Zero offset coeff only
        coeffs =  params[1:] + (1,)  # Add fixed 1 slope
    else:   
        coeffs =  params[1:]  # Use all the coefficents for transforming polynomial 

    # Overide any default params
    if defaultParamDic:
        coeffs = update_coeffs_with_defaults(coeffs,defaultParamDic)

    if method == 'p':
        Xtransformed = np.polynomial.polynomial.polyval(Xoriginal, coeffs)
    elif method == 'c':
        Xtransformed = np.polynomial.chebyshev.chebval(Xoriginal, coeffs)
    elif method == 'v': # (1+v/c) shift
        Xtransformed = Xoriginal*(1+coeffs[0]/speed_of_light)
    elif method == 'x': # (w + w v/c +P dw/dp) combined shift
        Xtransformed = Xoriginal*(1+coeffs[0]/speed_of_light) + coeffs[1]*np.gradient(Xoriginal)
    else:
        print('method {0} is not implemented'.format(method))
        return None
        
    # interpolate the original spectrum to new coordinates
    tck = interp.splrep(Xoriginal, scaledFlux)
    return interp.splev(Xtransformed, tck,ext=3)


def errorfunc_tominimise(params,method='l',Reg=0,RefSpectrum=None,DataToFit=None,sigma=None,defaultParamDic=None,**kwargs ):
    """ Error function to minimise to fit model.
    Currently implemented for only the regularised fitting of Legendre coefficent transform
    Reg is the Regularisation coefficent for LASSO regularisation.
    defaultParamDic: is the dictionary of the deafult values for all the parameters which can include parameters not being fitted.
                      For example: for method=l , defaultParamDic = {'v':0,'p':0,'w':0}"""
    
    # First paramete is the flux scaling
    scaledFlux = RefSpectrum*params[0]
    if method == 'l':
        grid = np.linspace(-1,1,len(RefSpectrum))
        if 'WavlCoords' in kwargs:
            Xoriginal = kwargs['WavlCoords']
        else:
            Xoriginal = None
        if 'LCRef' in kwargs:
            LCRef = kwargs['LCRef']
            if Xoriginal is None:
                Xoriginal = np.polynomial.legendre.legval(grid,LCRef) 
        else:
            if ('ldeg' in kwargs) and ('WavlCoords' in kwargs) :
                LCRef = np.polynomial.legendre.legfit(grid,Xoriginal,deg=ldeg)
        paramstring = kwargs['paramstofit']
        if defaultParamDic is None:
            PVWdic = {'v':0,'p':0,'w':0}
        else:
            PVWdic = defaultParamDic
        for i,s in enumerate(paramstring):
            PVWdic[s] = params[i+1]

        normP = False
        if ('p' in paramstring) and ('v' in paramstring):
            normP = True  # Break degeneracy with v by normalising pixel shift
        LCnew = TransformLegendreCoeffs(LCRef,PVWdic,normP=normP)

        Xtransformed = np.polynomial.legendre.legval(grid,LCnew) 
    else:
        print('method {0} is not implemented'.format(method))
        return None

    # interpolate the original spectrum to new coordinates
    tck = interp.splrep(Xoriginal, scaledFlux)
    PredictedSpectrum = interp.splev(Xtransformed, tck)
    if sigma is None:
        sigma=1
    return  np.concatenate(((PredictedSpectrum-DataToFit)/sigma,np.sqrt(Reg*np.abs(params[1:]))))

    

def ReCalibrateDispersionSolution(SpectrumY,RefSpectrum,method='p3',initial_guess=None,sigma=None,cov=False,Reg=0,defaultParamDic=None,scalepixel=True):
    """ Recalibrate the dispertion solution of SpectrumY using 
    RefSpectrum by fitting the relative drift using the input method.
    Input:
       SpectrumY: Un-calibrated Spectrum Flux array
       RefSpectrum: Wavelength Calibrated reference spectrum (Flux vs wavelegnth array:(N,2))
       method: (str, default: p3) the method used to model and fit the drift in calibration
       initial_guess: (list, optional) Optional initial guess of the coeffients for the distortion model
       sigma: See sigma arg of scipy.optimize.curve_fit ; it is the inverse weights for residuals
       cov: (bool, default False) Set cov=True to return an estimate of the covarience matrix of parameters 
       Reg: Regularisation parameter for LASSO (Currently implemented only for multi parameter Legendre polynomials) 
       defaultParamDic: Default values for parameters in a multi parameter model. Example for l* methods. 
       scalepixel: (bool, default True) scale input coordinates to -1 to 1, NOTE: Currently this scaling done only for method = p* and c*.
      Available methods: 
               pN : Fits a Nth order polynomial distortion  
               cN : Fits a Nth order Chebyshev polynomial distortion 
               v  : Fits a velocity redshift distortion
               x  : Fits a velocity redshift distortion and a 0th order pixel shift distortion 
               lwN: Fits Nth order Legendre coefficent transform with wavelenth shift as the single parameter 
               lvN: Fits Nth order Legendre coefficent transform with velocity shift as the single parameter
               lpN: Fits Nth order Legendre coefficent transform with pixel shift as the single parameter 
               lpwN: Fits Nth order Legendre coefficent transform with pixel shift and wavelenth shift as two parameters
               lvwN: Fits Nth order Legendre coefficent transform with velocity shift and wavelenth shift as two parameters
               lpvN: Fits Nth order Legendre coefficent transform with pixel shift and velocity shift as two parameters
               lpvwN: Fits Nth order Legendre coefficent transform with pixel shift, velocity shift, and wavelength shift as three parameters
    Returns:
        wavl_sln : Output wavelength solution
        fitted_drift : the fitted calibration drift coeffients 
                    (IMP: These coeffs is for the method and scaling done inside this function)
    """
    RefFlux = RefSpectrum[:,1]
    RefWavl = RefSpectrum[:,0]

    # For stability and fast convergence lets scale the wavelength to -1 to 1 interval. (Except for doppler shift method)
    if method[0] in ['p','c']:
        if scalepixel:
            scaledWavl = scale_interval_m1top1(RefWavl,a=min(RefWavl),b=max(RefWavl))
        else:
            scaledWavl = RefWavl

    if (method[0] == 'p') and method[1:].isdigit():
        # Use polynomial of p* degree.
        deg = int(method[1:])
        # Initial estimate of the parameters to fit
        # [scalefactor,*p] where p is the polynomial coefficents
        if initial_guess is not None:
            p0 = initial_guess
        elif deg > 0:
            p0 = [1,0,1]+[0]*(deg-1)
        else:
            p0 = [1,0]
        poly_transformedSpectofit = partial(transformed_spectrum,method='p',WavlCoords=scaledWavl,defaultParamDic=defaultParamDic)
        popt, pcov = optimize.curve_fit(poly_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        if deg < 1: # Append slope 1 coeff
            popt = np.concatenate([popt, [1]])

        # Overide any default params
        coeffs = popt[1:]
        if defaultParamDic:
            coeffs = update_coeffs_with_defaults(coeffs,defaultParamDic)

        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        transformed_scaledWavl = np.polynomial.polynomial.polyval(scaledWavl, coeffs)

    elif (method[0] == 'c') and method[1:].isdigit():
        # Use chebyshev polynomial of c* degree.
        deg = int(method[1:])
        # Initial estimate of the parameters to fit
        # [scalefactor,*c] where c is the chebyshev polynomial coefficents
        if initial_guess is not None:
            p0 = initial_guess
        elif deg > 0:
            p0 = [1,0,1]+[0]*(deg-1)
        else:
            p0 = [1,0]
        cheb_transformedSpectofit = partial(transformed_spectrum,method='c',WavlCoords=scaledWavl,defaultParamDic=defaultParamDic)
        popt, pcov = optimize.curve_fit(cheb_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        if deg < 1: # Append slope 1 coeff
            popt = np.concatenate([popt, [1]])

        # Overide any default params
        coeffs = popt[1:]
        if defaultParamDic:
            coeffs = update_coeffs_with_defaults(coeffs,defaultParamDic)

        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        transformed_scaledWavl = np.polynomial.chebyshev.chebval(scaledWavl, coeffs)

    elif (method[0] == 'v'):
        # Use the 1+v/c formula to shift the spectrum. Usefull in grating sectrogrpahs where flexure is before grating.
        # Initial estimate of the parameters to fit  [1 for scaling, and 100 for velocity]
        if initial_guess is not None:
            p0 = initial_guess
        else:
            p0 = [1,100]
        vel_transformedSpectofit = partial(transformed_spectrum,method='v',WavlCoords=RefWavl,defaultParamDic=defaultParamDic)
        popt, pcov = optimize.curve_fit(vel_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        wavl_sln = RefWavl *(1+ popt[1]/speed_of_light)

    elif (method[0] == 'x'):
        # Use the w+ w*v/c + deltaP*dw/dp formula to shift the spectrum. Usefull in grating sectrogrpahs where flexure is before grating as well as after grating.
        # Initial estimate of the parameters to fit  [1 for scaling, and 100 for velocity,0 for pixshift]
        if initial_guess is not None:
            p0 = initial_guess
        else:
            p0 = [1,100,0]
        velp_transformedSpectofit = partial(transformed_spectrum,method='x',WavlCoords=RefWavl,defaultParamDic=defaultParamDic)
        popt, pcov = optimize.curve_fit(velp_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        wavl_sln = RefWavl *(1+ popt[1]/speed_of_light) + popt[2]*np.gradient(RefWavl)


    elif (method[0] == 'l'):
        # Usefull in grating sectrogrpahs where flexure is before grating as well as after grating.
        # Parameters to fit
        if defaultParamDic is None:
            PVWdic = {'v':0,'p':0,'w':0}
        else:
            PVWdic = defaultParamDic

        paramstring = [s for s in method[1:] if not s.isdigit()]
        ldeg = int(''.join([s for s in  method[1:] if s.isdigit()]))

        # Legendre coefficents for the polynomial
        grid = np.linspace(-1,1,len(RefWavl))
        LCRef = np.polynomial.legendre.legfit(grid,RefWavl,deg=ldeg)
        
        Initp={'v':1e-6,'p':1e-6,'w':1e-3}
        # Initial estimate of the parameters to fit  [0 for each parameter to fit]
        if initial_guess is not None:
            p0 = initial_guess
        else:
            p0 = [1]+[Initp[s] for s in paramstring]  # 1 is for scaling, rest are the parameters
        x_scaledic = {'v':1e-7,'p':1e-6,'w':1e-3}  # Approximate scales of the parameters
        x_scale = [1.] + [x_scaledic[s] for s in paramstring]  # 1 is for scaling, rest are the parameters
        l_errorfunc_tominimise = partial(errorfunc_tominimise,method='l',Reg=Reg,paramstofit=paramstring,
                                         WavlCoords=RefWavl,RefSpectrum=RefFlux,DataToFit=SpectrumY,sigma=sigma,
                                         LCRef=LCRef,defaultParamDic=PVWdic) 
        fitoutput = optimize.least_squares(l_errorfunc_tominimise,p0,x_scale=x_scale,ftol=None,xtol=1e-10)
        popt = fitoutput['x'] 
        print('Fitting {0} terminated in status number {1}'.format(paramstring,fitoutput['status']))
        if cov :
            pcov = calculate_cov_matrix_fromscipylsq(fitoutput)
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        for i,s in enumerate(paramstring):
            PVWdic[s] = popt[i+1]

        normP = False
        if ('p' in paramstring) and ('v' in paramstring):
            normP = True  # Break degeneracy with v by normalising pixel shift
        LCnew = TransformLegendreCoeffs(LCRef,PVWdic, normP=normP)

        wavl_sln = np.polynomial.legendre.legval(grid,LCnew) 

    else:
        raise NotImplementedError('Unknown fitting method {0}'.format(method))

    if method[0] in ['p','c']:
        if scalepixel:
            wavl_sln = scale_interval_m1top1(transformed_scaledWavl,
                                             a=min(RefWavl),b=max(RefWavl),
                                             inverse_scale=True)
        else:
            wavl_sln = transformed_scaledWavl

    if cov :
        return wavl_sln, popt, pcov
    else:
        return wavl_sln, popt
        

def calculate_pixshift_with_phase_cross_correlation(shifted_spec,reference_spec,upsample_factor=10):
    """ Returns the pixel shift between `shifted_spec` and `reference_spec` at the resolution of 1/upsample_factor """
    shift = registration.phase_cross_correlation(reference_spec,shifted_spec,upsample_factor=upsample_factor)[0]
    if isinstance(shift,np.ndarray): # Support for latest skimage version
        return shift[0]
    else:
        return shift

def parse_args(raw_args=None):
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Non-Interactive Wavelength Re-Calibration Tool")
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the uncalibrated Spectrum Flux array")
    parser.add_argument('RefSpectrumFile', type=str,
                        help="Reference Spectrum file which is calibrated, containing Flux vs wavelengths for the same pixels")
    parser.add_argument('OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength array")
    args = parser.parse_args(raw_args)
    return args
    
def main(raw_args=None):
    """ Standalone Interactive Line Identify Tool """
    args = parse_args(raw_args)
    SpectrumY = np.load(args.SpectrumFluxFile)
    RefSpectrum = np.load(args.RefSpectrumFile)
    Output_fname = args.OutputWavlFile
    wavl_sln, fitted_drift = ReCalibrateDispersionSolution(SpectrumY,RefSpectrum,method='p3')
    np.save(Output_fname,wavl_sln)
    print('Wavelength solution saved in {0}'.format(Output_fname))

if __name__ == "__main__":
    main()
