#!/usr/bin/env python
""" This is a non-interactive tool to re-identify calibration lamp lines from 
an already identified dispersion file """
import os
import argparse
import numpy as np
from .iidentify import TryToFitNewLinesinSpectrum, get_fitted_function, read_dispersion_inputfile, load_fluxdata, write_wavldata
from scipy.constants import speed_of_light

def parse_args(raw_args=None):
    """ Parses the command line input arguments """
    epilog_str=""" Use {0} in the filename if multiple orders are fitted in a single run. """
    parser = argparse.ArgumentParser(description="Non-Interactive Wavelength Re-Calibration Tool",
                                     epilog=epilog_str)
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the Spectrum's Flux array")
    parser.add_argument('--fits_ext', type=int, default=0,
                        help="Extension to load if the input flux file is fits")
    parser.add_argument('--fits_ext_var', type=int,
                        help="Extension of the variance array for the input flux fits file (optional)")
    parser.add_argument('RefDispTableFile', type=str,
                        help="Reference file containing the table of Wavelengths, Pixels, Error")
    parser.add_argument('OutDispTableFile', type=str,
                        help="Output Filename to write the table of Wavelengths, Pixels, Error")
    parser.add_argument('--OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength solution array")
    parser.add_argument('--ModelForDispersion', type=str,default='l6',
                        help="Model to fit the individual line positions to obtain the full wavelength array dispersion solution")
    parser.add_argument('--SavePlots', action='store_true', 
                        help="Save plots as well of the fitted dispersion solution in the same filename `OutputWavlFile` with .png extension")
    parser.add_argument('--StackOrders', action='store_true', 
                        help="Save a stacked single wavength solution file for all orders")
    args = parser.parse_args(raw_args)
    if args.SavePlots and (args.OutputWavlFile is None):
        parser.error("--SavePlots requires --OutputWavlFile")
    if args.StackOrders and (args.OutputWavlFile is None):
        parser.error("--StackOrders requires --OutputWavlFile")
    return args
    
def main(raw_args=None):
    """ Standalone Non-Interactive Line Re-Identify Tool """
    args = parse_args(raw_args)    

    if args.SavePlots:
        import matplotlib.pyplot as plt

    SpectrumY_all = load_fluxdata(args.SpectrumFluxFile,fits_ext=args.fits_ext)
    if args.fits_ext_var is not None:
        SpectrumY_Var_all = load_fluxdata(args.SpectrumFluxFile,fits_ext=args.fits_ext_var)
    else:
        SpectrumY_Var_all = [None]*len(SpectrumY_all)

    Refdisp_fname = args.RefDispTableFile
    Outdisp_fname = args.OutDispTableFile
    if len(SpectrumY_all.shape) < 2:  # If the spectrum is a 1D single order spectrum
        SpectrumY_all = [SpectrumY]  # Pack it into a single element list
        SpectrumY_Var_all = [SpectrumY_Var_all]

    if args.StackOrders:
        CoeffDictionary_All = {}
        WavlSolutionArray_All = []

    for order, (SpectrumY, SpectrumY_Var) in enumerate(zip(SpectrumY_all,SpectrumY_Var_all)):
        print('Fitting Order: {0}'.format(order))
        if os.path.isfile(Outdisp_fname.format(order)):
            print('Output dispersion file {0} exists.'.format(Outdisp_fname.format(order)))
            print('Only non-calibrated wavelenghts in it will be re-fitted..')
        else:
            # Create the dispersion file with all the wavelengths in Reference file
            wavelengths_tofit, (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(Refdisp_fname.format(order))
            with open(Outdisp_fname.format(order),'w') as outdispfile:
                outdispfile.write('# Wavelengths Pixel Sigma # comments\n')
                for wavel in wavelengths_inp + wavelengths_tofit :
                    outdispfile.write('{0} # Re-Fitted\n'.format(wavel))

        plot_pdf_output = Outdisp_fname.format(order)+'_linefit_plots.pdf' if args.SavePlots else None
        # Now recalibrate the positions of the line by fitting lines again to new positions
        TryToFitNewLinesinSpectrum(SpectrumY,Outdisp_fname.format(order),LineSigma=1.5,
                                   reference_dispfile = Refdisp_fname.format(order), SpectrumY_Var=SpectrumY_Var,
                                   guess_function=args.ModelForDispersion, plot_pdf_output=plot_pdf_output)

        print('Dispersion ASCII input file saved in file://{0}'.format(Outdisp_fname.format(order)))

        if args.OutputWavlFile:
            # Also save the full Wavelength array solution
            Output_fname = args.OutputWavlFile

            # Read back the calibration
            _ , (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(Outdisp_fname.format(order))

            wavelengths_inp = np.array(wavelengths_inp) 
            # Scale the pixel to -1 to 1 range for stable polynomial function
            scaled_pixels_inp = (2.*np.array(pixels_inp)/(len(SpectrumY)-1.)) - 1
            # Dispertion function
            
            disp_func, coeffs, Mask = get_fitted_function(pixels=scaled_pixels_inp,
                                                          wavel=wavelengths_inp,
                                                          sigma=sigma_inp, 
                                                          method=args.ModelForDispersion,
                                                          return_coeff=True,
                                                          sigma_to_clip=3)
            if np.sum(~Mask):
                print('Rejected following oultliers by {0} sigma clipping in the fit'.format(3))
                print('\n'.join(map(str,np.array(wavelengths_inp)[~Mask])))
                print('*'*5)

            # Calculate the wavlength array
            wavl = disp_func(np.linspace(-1,1,len(SpectrumY)))

            # Calculate the residuals of the fit
            wavl_residue = wavelengths_inp - disp_func(scaled_pixels_inp)
            velocity_residue = speed_of_light*wavl_residue/wavelengths_inp
            CoeffDictionary = {}
            CoeffDictionary['CTYPE{0}'.format(order+1)] = ('WAVE-PLY' , 'Wavelength axis')
            CoeffDictionary['PS{0}_0'.format(order+1)] = (args.ModelForDispersion, 'Polynomial of dispersion solution')
            CoeffDictionary['PS{0}_1'.format(order+1)] = (True, 'Domain scaled from 1 to -1')
            CoeffDictionary.update({'PV{0}_{1}'.format(order+1,i):(c,'{0}th order Coeff'.format(i)) for i,c in enumerate(coeffs)})
            CoeffDictionary['SigmaW{0}'.format(order+1)] = (np.std(wavl_residue[Mask]), 'Sigma of Wavelength Residue')
            CoeffDictionary['SigmaV{0}'.format(order+1)] = (np.std(velocity_residue[Mask]), 'Sigma of Velocity Residue (m/s)')

            _ = write_wavldata(Output_fname.format(order),wavl,fits_headerDic=CoeffDictionary)
            print('Wavelength solution saved in {0}'.format(Output_fname.format(order)))

            if args.StackOrders:
                CoeffDictionary_All.update(CoeffDictionary)
                WavlSolutionArray_All.append(wavl)
            
            if args.SavePlots:
                Output_plot_fname = os.path.splitext(Output_fname.format(order))[0]+'.png'
                fig = plt.figure(figsize=(16,8))
                ax1 = plt.subplot(211)
                ax2 = plt.subplot(212, sharex = ax1)
                ax1.plot(scaled_pixels_inp,wavelengths_inp,'.',color='k',label='Calibration Lines')
                ax1.plot(np.linspace(-1,1,len(SpectrumY)),wavl,'-',color='orange',label='Dispersion Solution: {0}'.format(args.ModelForDispersion))
                ax1.plot(scaled_pixels_inp[~Mask],wavelengths_inp[~Mask],'x',color='r',label='Outliers')
                ax1.legend()
                ax1.set_ylabel('Wavelength')
                plt.title('Order {0}'.format(order))
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax2.axhline(y=0,color='k',ls='--',alpha=0.7)
                ax2.plot(scaled_pixels_inp[Mask],velocity_residue[Mask],'.',color='k',label='Residue') 
                # ax2.plot(scaled_pixels_inp[~Mask],wavl_residue[~Mask],'x',color='r',label='Outliers') 
                ax2.text(0,0,'Sigma V = {0:.2e} m/s'.format(CoeffDictionary['SigmaV{0}'.format(order+1)][0]),color='blue')
                ax2.legend()
                ax2.set_ylabel('Residue in Velocity (m/s)')
                ax2.set_xlabel(r'Pixels (scaled to -1 to 1)')
                plt.minorticks_on()
                plt.tick_params(pad=4)
                plt.ticklabel_format(useOffset=False)
                fig.subplots_adjust(hspace=0)
                print(Output_plot_fname)
                fig.savefig(Output_plot_fname)
                print('Saved plot to {0}'.format(Output_plot_fname))
                plt.close()
                
    if args.StackOrders:
        _ = write_wavldata(Output_fname.format('all'),np.array(WavlSolutionArray_All),fits_headerDic=CoeffDictionary_All)
        print('Stacked wavelength solution saved in {0}'.format(Output_fname.format('all')))
        

if __name__ == "__main__":
    main()
