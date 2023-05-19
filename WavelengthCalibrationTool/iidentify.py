#!/usr/bin/env python
""" This is an interactive tool to identify calibration lamp lines from line atlas """
from .utils import NearestIndex, FitLineToData
import sys
import shutil
import uuid
import time
import readline
import argparse
import re
import os
from multiprocessing import Process, Pipe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.stats import sigma_clip
from termios import tcflush, TCIOFLUSH
import scipy.interpolate

# For python 2, replace the input function with raw_input
if sys.version_info < (3,0):
    input = raw_input

def SelectLinesInteractively(SpectrumX,SpectrumY,ExtraUserInput=None,comm_pipe=None, LineSigma=1.5):
    """ Fits Gaussian to all points user press m and c to confirm
    ExtaUserInput : (str) If provided will ask user from the extra input for each line 
                        Example: 'Wavelength'

    comm_pipe : One end of the Pipe which needs to be used for getting input from parent process.
    LineSigma : Approximated Sigma width of the line in pixels. 
                Size of the window region to use for fitting line will be 5*LineSigma.
    Returns : List of Fitted Line Models (and user input if any).

    IMP: This should always be run as a seperate Process.
    """
    print('Close the poped up plot window after done with selection')
    figi = plt.figure()
    ax = figi.add_subplot(111)
    ax.set_title('Press m to select line, then c to confirm the selection.')
    ax.plot(SpectrumX,SpectrumY,linestyle='--', drawstyle='steps-mid',marker=',',color='k')
    junk, = ax.plot([],[])
    PlotedLines=[junk]
    LinesConfirmedToReturn = []
    LatestLineModel = [None]
    WindowRange = [-9e9, -9.1e9]  # Default window range
    # Define the function to run when a span is selected
    def onselect(xmin, xmax):
        WindowRange[0] = xmin
        WindowRange[1] = xmax
        print('Updated Window range to {0}. Width: {1}'.format(WindowRange,xmax-xmin))

    # Define the function to run while key is pressed
    def on_key(event):
        if event.key == 'm' :
            Xpos = event.xdata
            if WindowRange[0] < Xpos <  WindowRange[1]:
                print('Fitting inside Window {0}'.format(WindowRange))
                Model_fit = FitLineToData(SpectrumX,SpectrumY,Xpos,event.ydata,
                                          AmpisBkgSubtracted=False,Sigma = LineSigma,
                                          WindowStartEnd = WindowRange)
            else:
                print('Fitting using 2*5*{0} pixel Window'.format(LineSigma))
                Model_fit = FitLineToData(SpectrumX,SpectrumY,Xpos,event.ydata,
                                          AmpisBkgSubtracted=False,Sigma = LineSigma,
                                          WindowStartEnd = None)
            if PlotedLines:
                ax.lines.remove(PlotedLines[-1])  # Remove the last entry in the plotedlines
            IndxCenter = NearestIndex(SpectrumX,Model_fit.mean_0.value)
            SliceToPlotX = SpectrumX[max(0,int(IndxCenter-5*LineSigma)):min(int(IndxCenter+5*LineSigma+1),len(SpectrumX)+1)]
            linefit, =  ax.plot(SliceToPlotX,Model_fit(SliceToPlotX),color='r')
            PlotedLines.append(linefit)
            ax.figure.canvas.draw()

            LatestLineModel[0] = Model_fit
            print(Model_fit)


        elif event.key == 'c' :
            print('Fit acceptence confirmed.')
            junk, = ax.plot([],[])
            PlotedLines.append(junk)  # Add a new junk plot to end of ploted lines

            if ExtraUserInput is not None:  # Ask user for the extra input
                # Send the Input msg string and the model fit value
                comm_pipe.send('{0}: |{1}'.format(ExtraUserInput,LatestLineModel[0].mean_0.value))
                Uinput = comm_pipe.recv()
                LinesConfirmedToReturn.append((LatestLineModel[0],Uinput))
            else:
                LinesConfirmedToReturn.append(LatestLineModel[0])
            

    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))

    cid = figi.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    comm_pipe.send('')
    # print('Identified Lines : {0}'.format(LinesConfirmedToReturn))
    return LinesConfirmedToReturn

def get_fitted_function(pixels,wavel,sigma=None,method='c3',return_coeff=False,sigma_to_clip=False):
    """ Returns the fitted function f(pixels) = wavel .
    Define all the methods to use for fitting here
    pX : polynoial of X degree
    cX : Chebyshev polynoial of X degree
    lX : Legendre polynoial of X degree
    sX : spline with smoothing parameter X
    """
    output_object = None
    pixels = np.array(pixels)
    wavel = np.array(wavel)
    sigma = np.array(sigma)

    if (method[0] == 'p') and method[1:].isdigit():
        # Use polynomial of p* degree.
        deg = int(method[1:])
        p,residuals,rank,sing_values,rcond = np.polyfit(pixels,wavel,deg,w=1/sigma,full=True)
        print('Stats of the poly fit of degree {0}'.format(deg))
        print('Poly Coeffs p: {0}'.format(p))
        print('residuals:{0},  rank:{1}, singular_values:{2}, rcond:{3}'.format(residuals,
                                                                                rank,sing_values,
                                                                                rcond))
        output_object = np.poly1d(p)
        coeffs = p

    elif (method[0] == 'c') and method[1:].isdigit():
        # Use Chebyshev polynomial of c* degree.
        deg = int(method[1:])
        c, (residuals,rank,sing_values,rcond) = np.polynomial.chebyshev.chebfit(pixels,wavel,deg,w=1/sigma,full=True)
        print('Stats of the Chebyshev polynomial fit of degree {0}'.format(deg))
        print('Cheb Coeffs c: {0}'.format(c))
        print('residuals:{0},  rank:{1}, singular_values:{2}, rcond:{3}'.format(residuals,
                                                                                rank,sing_values,
                                                                                rcond))
        output_object = lambda x : np.polynomial.chebyshev.chebval(x, c)
        coeffs = c

    elif (method[0] == 'l') and method[1:].isdigit():
        # Use Legendre polynomial of l* degree.
        deg = int(method[1:])
        c, (residuals,rank,sing_values,rcond) = np.polynomial.legendre.legfit(pixels,wavel,deg,w=1/sigma,full=True)
        print('Stats of the Legendre polynomial fit of degree {0}'.format(deg))
        print('Legendre Coeffs c: {0}'.format(c))
        print('residuals:{0},  rank:{1}, singular_values:{2}, rcond:{3}'.format(residuals,
                                                                                rank,sing_values,
                                                                                rcond))
        output_object = lambda x : np.polynomial.legendre.legval(x, c)
        coeffs = c

    elif (method[0] == 's'):
        # Use spline with smoothing parameter s*
        if method[1:]:
            smooth = float(method[1:])
        else:
            smooth = None
        isort = np.argsort(pixels)  # indices for sorting the input to spline funtion
        output_object = scipy.interpolate.UnivariateSpline(pixels[isort],wavel[isort], w=1/sigma[isort],k=3, s=smooth, ext=0)
        print('Stats of the Spline fit of smoothness {0}'.format(smooth))
        print('Spline Knots :{0}'.format(output_object.get_knots()))
        print('Spline Coeffs :{0}'.format(output_object.get_coeffs()))
        print('residuals:{0}'.format(output_object.get_residual()))
        coeffs = output_object.get_coeffs()

    else:
        print('Error: unknown fitting method {0}'.format(method))
        raise NotImplementedError

    if sigma_to_clip:
        # Do a sigma clipping filtering of the data points
        residue = (wavel - output_object(pixels))/sigma
        filtered_residue = sigma_clip(residue, sigma=sigma_to_clip)
        Mask = ~filtered_residue.mask
        print('Refitting after rejecting {0} outliers'.format(np.sum(~Mask)))
        output_object, coeffs = get_fitted_function(pixels[Mask],wavel[Mask],sigma=sigma[Mask],method=method,return_coeff=True,sigma_to_clip=False)

    if return_coeff and sigma_to_clip:
        return output_object, coeffs, Mask
    elif return_coeff:
        return output_object, coeffs
    else:
        return output_object

def read_dispersion_inputfile(filename):
    """ Reads the Dispersion Input text file.
    Returns a list of wavelengths without pixel ref
    And a tuple of (wavelengths, pixels, sigma) containing lists"""
    wavelengths_without_pixels = []
    wavelengths = []
    pixels = []
    sigma = []

    try :
        with open(filename) as f:
            for line in f:
                if line[0] == '#' : continue  # commented lines
                line = line.rstrip().split('#')[0] # remove any inline comments
                if len(line.split()) == 1:
                    wavelengths_without_pixels.append(float(line))
                elif len(line.split()) == 3:
                    wavelengths.append(float(line.split()[0]))
                    pixels.append(float(line.split()[1]))
                    sigma.append(float(line.split()[2]))
                elif len(line.split()) == 2:
                    wavelengths.append(float(line.split()[0]))
                    pixels.append(float(line.split()[1]))
                    sigma.append(1) # default 1 pix as sigma
    except IOError:
        print('ERROR: Cannot read dispersion file: {0}'.format(filename))
        raise

    return wavelengths_without_pixels, (wavelengths,pixels,sigma)

def writeto_dispersion_inputfile(filename,wavelengths,pixels,sigma,backupsuffix='.bak'):
    """ writes the fitted pixel positions to inputfile .
    Oldfile is backedup with .bak prefix"""
    # First backup file # Old backup will be overwritten
    shutil.move(filename,filename+backupsuffix)
    
    already_addedinfile = []
    with open(filename+backupsuffix,'r') as oldfile, open(filename,'w') as newfile:
        for line in oldfile:
            try:
                wavel = float(line.rstrip().split()[0])
            except ValueError:
                pass
            else:
                if wavel in wavelengths:
                    ind = wavelengths.index(wavel)
                    textindx = len(line.rstrip().split()[0])
                    line = line[:textindx] +\
                           ' {0} {1} #'.format(pixels[ind],sigma[ind]) +\
                           line[textindx:]
                    already_addedinfile.append(wavel)
            finally:
                newfile.write(line)

        # Now append any remaining new lines
        for wavel,pix,w in zip(wavelengths,pixels,sigma):
            if wavel not in already_addedinfile:
                newfile.write('{0} {1} {2}\n'.format(wavel,pix,w))

def update_main_figure(fig_main,SpectrumY,wavltofit__wavl_pix_sigma):
    """ Updates the main plot of the latest dispersion fit.
    Also returns the latest dispersion function used"""
    print(wavltofit__wavl_pix_sigma[1])
    wavelengths_inp,pixels_inp,sigma_inp = wavltofit__wavl_pix_sigma[1]
    wavelengths_tofit = wavltofit__wavl_pix_sigma[0]
    disp_func = get_fitted_function(pixels=pixels_inp,
                                    wavel=wavelengths_inp,
                                    sigma=sigma_inp, method='c6')
    fig_main.clf()
    ax = fig_main.add_subplot(111)

    # Plot all the fitted lines as well as unfitted lines
    for xc in wavelengths_inp: 
        ax.axvline(x=xc, color='g', linestyle='--',alpha=0.7)
    for xc in wavelengths_tofit: 
        ax.axvline(x=xc, color='b', linestyle=':',alpha=0.3)
    # Plot the spectrum
    ax.plot(disp_func(np.arange(len(SpectrumY))),SpectrumY,
            linestyle='-', drawstyle='steps-mid',color='k') #marker='.'

    #Colormap which is red when maximum
    colormap = plt.get_cmap('PuRd')
    pointYlevel = np.median(SpectrumY)
    residuesW = [np.abs(w-disp_func(pix))/np.abs(disp_func(pix)-disp_func(pix+sig)) \
                 for w,pix,sig in zip(wavelengths_inp,pixels_inp,sigma_inp)]

    NresiduesW = np.array(residuesW)/max(residuesW)
    print('RMS of residue: {0}'.format(np.sqrt(np.mean(np.power(residuesW,2)))))
    print('RMS of Normalised residue: {0}'.format(np.sqrt(np.mean(np.power(NresiduesW,2)))))
    print('Individual Normalised Residues of fit:')
    print('Wavel  Pixel  Nresidue')
    for w,pix,Nresid in zip(wavelengths_inp,pixels_inp,NresiduesW):
        ax.plot([w,disp_func(pix)],[pointYlevel]*2,color=colormap(int(Nresid*255)))
        ax.plot([w],[pointYlevel],'.',color=colormap(int(Nresid*255)))
        print('{0}   {1}  {2}'.format(w,pix,Nresid))

    print('>>')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Counts')
    fig_main.show()
    return disp_func

def AddlinesbyInteractiveSelection(SpectrumY,disp_filename,comm_pipe=None):
    """ Adds more data points to disp_filename by interactively selecting lines """
    print('Enter more point in the file, or select lines '
          'interatively to obtain Zeroth order solution')
    ListOfLines = SelectLinesInteractively(np.arange(len(SpectrumY)),
                                           SpectrumY,
                                           ExtraUserInput='Wavelength',
                                           comm_pipe=comm_pipe)
    wavl,pix,sigma = [] , [], []
    for linefit in ListOfLines:
        wavl.append(float(linefit[1]))
        pix.append(linefit[0].mean_0.value)
        sigma.append(1)
    if wavl:
        writeto_dispersion_inputfile(disp_filename,wavl,pix,sigma)


def TryToFitNewLinesinSpectrum(SpectrumY,disp_filename,LineSigma=1.5,reference_dispfile = None, 
                               SpectrumY_Var = None, guess_function='c5', plot_pdf_output = None):
    """ Try to fit gaussian at the line wavelengths without pixel position in the disp_filename.
    And add the fitted pixels values to the file.
    If reference_dispfile is provided, data in that file will be used to claculate dispersion solution
    SpectrumY_Var (optional): Variance of the SpectrumY array if need to be considerd in line fitting.
    guess_function (default='c5'): Polynomial model to use to interpolate existing solution for initial guess of new line positions.
    plot_pdf_output (optional): Save the plots of the individual line fits in a multipage pdf file is a filename is provided.
    """
    if reference_dispfile is None:  # use the entry in disp_filename itself
        reference_dispfile = disp_filename

    # First read the existing calibration
    _ , (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(reference_dispfile)
    # Now read the wavelengths to calibrate
    wavelengths_tofit, (_discard1,_discard2,_discard3) = read_dispersion_inputfile(disp_filename)
    # Dispertion function
    disp_func = get_fitted_function(pixels=pixels_inp,
                                    wavel=wavelengths_inp,
                                    sigma=sigma_inp, method=guess_function)

    pix_fitted = []
    sigma_fitted = []
    if plot_pdf_output is not None:
        pdfplots = PdfPages(plot_pdf_output)
    for wavel in wavelengths_tofit:
        XPixarray = np.arange(len(SpectrumY))
        Xpos = NearestIndex(disp_func(XPixarray),wavel)
        Lstart = int(np.rint(Xpos-3*LineSigma))
        Lend = int(np.rint(Xpos+3*LineSigma))+1
        Ampl_init = np.max(SpectrumY[Lstart:Lend]) - \
                    np.min(SpectrumY[Lstart:Lend])
        # Fit a Guassian line
        Model_fit = FitLineToData(XPixarray,SpectrumY,Xpos,Ampl_init,
                                  AmpisBkgSubtracted=False,Sigma = LineSigma, SpecY_Var = SpectrumY_Var)
        pix_fitted.append(Model_fit.mean_0.value)
        sigma_fitted.append(1) # to be updated later with actual error
        if plot_pdf_output is not None:
            # Make the plots to save to the pdf file
            fig = plt.figure(figsize=(6, 6))
            plt.plot(XPixarray[Lstart:Lend],SpectrumY[Lstart:Lend],'.',color='k')
            if SpectrumY_Var is not None:
                plt.errorbar(XPixarray[Lstart:Lend],SpectrumY[Lstart:Lend],yerr=np.sqrt(SpectrumY_Var[Lstart:Lend]),color='k')
            plt.plot(np.linspace(Lstart,Lend,(Lend-Lstart)*10),Model_fit(np.linspace(Lstart,Lend,(Lend-Lstart)*10)),color='r',label=str(Model_fit.mean_0.value))
            plt.title('Wavl = {0}'.format(wavel))
            plt.legend()
            pdfplots.savefig()
            plt.close()
    if plot_pdf_output is not None:
        pdfplots.close()
    # Write the fitted pixel positions
    if pix_fitted:
        writeto_dispersion_inputfile(disp_filename,wavelengths_tofit,pix_fitted,sigma_fitted)


def DisplayDispersionSolution(SpectrumY,disp_filename,comm_pipe=None):
    """ This keeps a window open with the plot of the latest dispersion solution.
    IMP: Needs to be run as a Process, and comminicated via Pipe """
    fig_main = plt.figure()
    ax = fig_main.add_subplot(111)
    ax.set_title('Press r to refresh the plot.')
    # Define the function to run while key is pressed
    disp_func_holder = [update_main_figure(fig_main,SpectrumY,read_dispersion_inputfile(disp_filename))]
    def on_key(event):
        if event.key == 'r' :
            disp_func_holder[0] = update_main_figure(fig_main,SpectrumY,read_dispersion_inputfile(disp_filename))
    cid = fig_main.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    disp_func = disp_func_holder[0]
    if disp_func is not None:
        comm_pipe.send(tuple((disp_func(np.arange(len(SpectrumY))), disp_filename)))
    else:
        comm_pipe.send(tuple((None, disp_filename)))
    

def StartInteractiveLineSelectionSubrocess(SpectrumY,disp_filename):
    """ Starts a subprocess for Adding lines interactively """
    parent_conn, child_conn = Pipe()
    fp = Process(target=AddlinesbyInteractiveSelection,args=(SpectrumY,disp_filename,child_conn))
    fp.start()
    Clientmsg = '>>'
    wavelengths_tofit, (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(disp_filename)
    FullWavelengthArray = np.array(wavelengths_tofit + wavelengths_inp)

    # if atlest 2 wavelegnths are alredy calibrated, we can predict default value form the line list
    disp_func = None
    if len(wavelengths_inp) > 1:
        if len(wavelengths_inp) == 2 : 
            pdeg = '1'
        elif len(wavelengths_inp) == 3 : 
            pdeg = '2'
        elif len(wavelengths_inp) > 20 : 
            pdeg = '6'
        else:
            pdeg = '3'
        disp_func = get_fitted_function(pixels=pixels_inp,
                                        wavel=wavelengths_inp,
                                        sigma=sigma_inp, method='c'+pdeg)

    while Clientmsg:
        Clientmsg = parent_conn.recv()
        if Clientmsg:
            PromptMsg = '|'.join(Clientmsg.split('|')[:-1])
            DefaultValue = ''
            try:
                FittedValue = float(Clientmsg.split('|')[-1])
            except ValueError:
                # No fitted value provided to find default value..
                pass
            else:
                if disp_func is not None:
                    DefaultValue = str(FullWavelengthArray[NearestIndex(FullWavelengthArray,disp_func(FittedValue))])
                    PromptMsg = PromptMsg+'(Default:{0}):'.format(DefaultValue)
                    
            tcflush(sys.stdin, TCIOFLUSH) # Flush anythin in terminal buffer
            usr_inp = input(PromptMsg)
            if usr_inp:
                parent_conn.send(usr_inp)
            else:
                parent_conn.send(DefaultValue)
    fp.join()
    
    
def InteractiveDispersionSolution(SpectrumY,disp_filename=None):
    """ Identify lines in the spectrum and find dispersion solution"""
    # First read the disp file
    if disp_filename is None:
        disp_filename = str(uuid.uuid4())
        with open(disp_filename,'w') as f:
            f.write('# Wavel Pixel Sigma \n')
    print('Open the file://{0} in a non-blocking text editor to edit'.format(disp_filename))

    wavelengths_tofit, (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(disp_filename)

    if len(wavelengths_inp) < 2:
        print('Not even two starting points to intiailise; Starting Interactive fit Window')
        StartInteractiveLineSelectionSubrocess(SpectrumY,disp_filename)
    # Plot main figure
    Mainparent_conn, Mainchild_conn = Pipe()
    Mp = Process(target=DisplayDispersionSolution,args=(SpectrumY,disp_filename,Mainchild_conn))
    Mp.start()
    DoneWithFitting = False
    print('Edit the file://{0} to add or remove points to fit'.format(disp_filename))
    print('Enter "done" to exit the interactive fitting')
    print('Enter f to start an interative fit window')
        
    while not DoneWithFitting:
        tcflush(sys.stdin, TCIOFLUSH) # Flush anythin in terminal buffer
        usr_input = input('>> ')
        if usr_input == 'done':
            print('Exiting the fitting tool..')
            print('Close all Figure windows to exit..')
            DoneWithFitting = True
        elif usr_input == 'f':
            print('Starting Interactive fit Window')
            StartInteractiveLineSelectionSubrocess(SpectrumY,disp_filename)
        elif usr_input == 'l':
            print('Fitting lines which do not have pixel coordinates in the text file.')
            TryToFitNewLinesinSpectrum(SpectrumY,disp_filename)

        
    wavel_soln_n_disp_filename =  Mainparent_conn.recv()
    Mp.join()
    # Return the dispertion wavelength solution and disp_filename 
    return wavel_soln_n_disp_filename
            
    

def parse_args(raw_args=None):
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Interactive Wavelength Calibration Tool")
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the Spectrum's Flux array")
    parser.add_argument('DispTableFile', type=str,
                        help="File containing the table of Wavelengths, Pixels, Error")
    parser.add_argument('OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength array")
    args = parser.parse_args(raw_args)
    return args

def load_fluxdata(filename,fits_ext=0):
    """ Loads the flux data """
    if os.path.splitext(filename)[-1] == '.npy':
        return np.load(filename)
    elif os.path.splitext(filename)[-1] == '.fits':
        return fits.getdata(filename,ext=fits_ext)
    else:
        print('Unknown input file extension for file : {0}'.format(filename))
        sys.exit(1)

def write_wavldata(output_filename,data,fits_headerDic=None):
    """ Writes the wavl solution data """
    if os.path.splitext(output_filename)[-1] == '.npy':
        np.save(output_filename,data)
    elif os.path.splitext(output_filename)[-1] == '.fits':
        hdu = fits.PrimaryHDU(data)
        if fits_headerDic is not None:
            for key,value in fits_headerDic.items():
                hdu.header[key] = value
        hdu.writeto(output_filename)
    else:
        print('Unknown output file extension for file : {0}'.format(filename))
        sys.exit(1)
    return output_filename

def main(raw_args=None):
    """ Standalone Interactive Line Identify Tool """
    args = parse_args(raw_args)
    extbracket = re.search('\[(\d+)\]$', os.path.splitext(args.SpectrumFluxFile)[-1])
    if extbracket is None:
        SpectrumY = load_fluxdata(args.SpectrumFluxFile)
    else:
        order = int(extbracket.group(1))
        SpectrumY = load_fluxdata(args.SpectrumFluxFile[:-len('[{0}]'.format(order))])[order]

    disp_fname = args.DispTableFile
    Output_fname = args.OutputWavlFile
    wavl,df = InteractiveDispersionSolution(SpectrumY,disp_filename=disp_fname)
    print('Dispersion ASCII input file saved in file://{0}'.format(df))
    np.save(Output_fname,wavl)
    print('Wavelength solution saved in {0}'.format(Output_fname))

if __name__ == "__main__":
    main()
