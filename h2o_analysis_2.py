import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import astropy.units as u
from astropy.io import fits  # We use fits to open the actual data file
from spectral_cube import SpectralCube
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
import matplotlib.patches as patches
import csv
import matplotlib.style as style
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import warnings
warnings.filterwarnings("ignore")
style.use('seaborn-v0_8-poster')
#plt.rcParams["font.family"] = 'serif'

# region 0) Automatic region detection for listings and definition of curveFit_peaks function
# Command to automatically update line numbers for the document
with open("LineNumber.py") as f:
    exec(f.read())

# figure for later use
fig5 = plt.figure(figsize=(15, 10), constrained_layout=True)

# Function definition
def curveFit_peaks(fresidual_spectrum, foriginal_spectrum, fpeak_width_pixel, fnumber_of_peaks, fwavelength_spectrum):
    fmaximum_index = np.argsort(-fresidual_spectrum)[:50]
    m = 0
    while m < len(fmaximum_index) - 1:
        n = m + 1
        while n < len(fmaximum_index):
            if abs(fmaximum_index[m] - fmaximum_index[n]) <= fpeak_width_pixel*2:
                fmaximum_index = np.delete(fmaximum_index, n)
            else:
                n += 1
        m += 1

    fmaximum_index = np.delete(fmaximum_index, np.s_[fnumber_of_peaks:len(fmaximum_index)])
    mask = np.full(fresidual_spectrum.shape, True)
    for index in fmaximum_index:
        mask[index - fpeak_width_pixel:index + fpeak_width_pixel + 1] = False

    # New curve fit
    new_pixel_fitted_polymodel = pixel_linfit(pixel_polymodel, fwavelength_spectrum[mask], foriginal_spectrum[mask])
    # Residual data with fitted curve excluding the highest peaks
    fnew_residual_spectrum = foriginal_spectrum - new_pixel_fitted_polymodel(fwavelength_spectrum)
    return fnew_residual_spectrum

# endregion 0)

# region 1) Define input parameters and load files
## Input parameters
# Area to be analysed
ra_mid_E1 = 25        # X Coordinate of the area's midpoint
dec_mid_E1 = 17       # Y Coordinate of the area's midpoint
ra_mid_E2 = 23        # X Coordinate of the area's midpoint
dec_mid_E2 = 13       # Y Coordinate of the area's midpoint
ra_span = 7           # Right Ascension area width
dec_span = 5          # Declination area height

# Define wavelength range
h2o_wav_min = 2.62    # Minimum value is 2.419
h2o_wav_max = 2.72    # Maximum value is 3.175

# Filter options
filter_data = False   # True if data should be filtered
filter_value = 200    # Define elimination percentage

# Range size and peak width for curve fit
nr_peaks = 8
peak_pixel = 2
peak_width = 0.0016 * u.micrometer


## Load files
# Get names of all datafiles
directory = r"C:\Users\steph\Stephan\KTH\Master_Thesis\02_Code\Data"
filelist = [r"jw01250006001_03103_00001_nrs2_s3d.fits",
            r"jw01250006001_03103_00002_nrs2_s3d.fits"]

# endregion 1)

for filename in filelist:

    # region 2) Define are, read data, filter error values and create subcubes with filtered area/range

    if filename == r"jw01250006001_03103_00001_nrs2_s3d.fits":
        ra_mid = ra_mid_E1
        dec_mid = dec_mid_E1
    else:
        ra_mid = ra_mid_E2
        dec_mid = dec_mid_E2

    # Range calculations
    ra_range = (ra_mid - ra_span, ra_mid + ra_span + 1)
    dec_range = (dec_mid - dec_span, dec_mid + dec_span + 1)
    wav_range = (h2o_wav_min, h2o_wav_max) * u.micrometer

    # Load the FITS file as a SpectralCube object
    filepath = directory + r"/" + filename
    hdul = fits.open(filepath)
    SCI_data = hdul[1]
    error = hdul[2].data
    cube = SpectralCube.read(SCI_data)
    size = SCI_data.header['PIXAR_SR']

    # Create sub-cubes with filtered areas and ranges
    cube_full = cube
    cube_area = cube.subcube(ylo=dec_range[0], yhi=dec_range[1], xlo=ra_range[0], xhi=ra_range[1])
    cube_rang = cube.spectral_slab(wav_range[0], wav_range[1])
    cube_raar = cube_area.spectral_slab(wav_range[0], wav_range[1])
    # cube_list = [cube_full, cube_area, cube_rang, cube_raar]
    cube_list = [cube_rang, cube_raar]

    # endregion 2)

    # region 3) Create the residual cubes with a curve fit, filter data and extract wavelengths

    # Extract the wavelengths from each cube and normalise each pixel value to create residual cubes
    wave_list = []
    residual_list = []
    residual_list_fit = []
    for i in range(0, len(cube_list)):

        # Filtering
        if filter_data:
            cube_list[i] = cube_list[i].with_mask(cube_list[i] < filter_value * 1e6 * u.Jy / u.steradian)

        wave_list.append(cube_list[i].spectral_axis)
        residual_list.append(np.zeros((cube_list[i].shape[0], cube_list[i].shape[1], cube_list[i].shape[2])))

    # Create residual cubes
    for j in range(0, cube_full.shape[1]):
        for k in range(0, cube_full.shape[2]):

            for residual_data in residual_list:
                if j < residual_data.shape[1] and k < residual_data.shape[2]:

                    i = residual_list.index(residual_data)

                    # Extract spectrum for single pixel on position (k, j)
                    pixel_spectrum = cube_list[i][:, j, k]
                    pixel_spectrum[np.isnan(pixel_spectrum)] = 0

                    # Set up curve model
                    pixel_polymodel = Polynomial1D(degree=5)
                    pixel_linfit = LinearLSQFitter()
                    pixel_fitted_polymodel = pixel_linfit(pixel_polymodel, wave_list[i], pixel_spectrum)
                    pixel_resi = pixel_spectrum - pixel_fitted_polymodel(wave_list[i])

                    # Second iteration of curve fit
                    residual_data[:, j, k] = curveFit_peaks(pixel_resi, pixel_spectrum, peak_pixel, nr_peaks,
                                                            wave_list[i])

                    # Filtering
                    if filter_data and np.any(residual_data[:, j, k] > filter_value):
                        residual_data[:, j, k][residual_data[:, j, k] > filter_value] = 0

    # endregion 3)

    # region 4) Fill the residual cubes and extract data from each cube

    cube_resi_list = []
    spec_list = []
    spec_resi_list = []
    wave_resi_list = []
    intensity_list = []
    intensity_resi_list = []
    max_inde = []
    max_spec = []
    max_wave = []

    for i in range(0, len(cube_list)):

        # Create the residual cubes
        cube_resi_list.append(SpectralCube(data=residual_list[i], wcs=cube_list[i].wcs, meta=cube_list[i].meta))

        # Extract the spectra, residual spectra and residual wavelengths for each cube
        spec_list.append(cube_list[i].mean(axis=(1, 2)))
        spec_resi_list.append(cube_resi_list[i].mean(axis=(1, 2)))
        spec_list[i][np.isnan(spec_list[i])] = 0
        spec_resi_list[i][np.isnan(spec_resi_list[i])] = 0
        wave_resi_list.append(cube_resi_list[i].spectral_axis.to(u.um))

        # Compute the average intensity values and average residual intensity values
        intensity_list.append(cube_list[i].sum(axis=0) / len(cube_list[i]))
        intensity_resi_list.append(cube_resi_list[i].sum(axis=0) / len(cube_resi_list[i]))
        intensity_list[i][np.isnan(intensity_list[i])] = 0
        intensity_resi_list[i][np.isnan(intensity_resi_list[i])] = 0

        # Find the maximum peaks for the residual spectra
        max_indices = np.argsort(-spec_resi_list[i])[:20]
        j = 0
        while j < len(max_indices) - 1:
            k = j + 1
            while k < len(max_indices):
                if abs(max_indices[j] - max_indices[k]) <= 3:
                    max_indices = np.delete(max_indices, k)
                else:
                    k += 1
            j += 1

        max_indices = np.delete(max_indices, np.s_[nr_peaks:len(max_indices)])
        max_indices = sorted(max_indices)
        max_inde.append(max_indices)
        max_spec.append(spec_resi_list[i][max_indices])
        max_wave.append(wave_resi_list[i][max_indices])

    # endregion 4)

    # region 5) Plot the results

    # Preparation
    plt.rcParams["font.family"] = 'serif'
    intensity_list_combined = intensity_list + intensity_resi_list
    cube_list_combined = cube_list + cube_resi_list
    axisfontsize = 16
    titlefontsize = 18
    figure_size = (15, 10)
    figure_label = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
    save_area = ''
    res = ''

    for i in range(0, len(spec_list)):

        #####################################################
        # Plot 1, 3: 2D spectral maps for full area and filtered area
        fig1 = plt.figure(figsize=figure_size, constrained_layout=True)
        ax1 = fig1.add_subplot(121, projection=intensity_list_combined[i * 2].wcs)
        im1 = ax1.imshow(intensity_list_combined[i * 2].hdu.data, cmap='cividis')
        ax1.set_xlabel("Right Ascension [hh:mm:ss]")
        ax1.set_ylabel("Declination [deg]", labelpad=-2)
        ax1.text(0.02, 0.98, figure_label[0], transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='top', color='w')
        cbar = plt.colorbar(im1, orientation='horizontal')
        cbar.set_label("Surface Brightness [MJy/sr]", fontsize=axisfontsize)
        rect = patches.Rectangle((ra_range[0] - 0.5, dec_range[0] - 0.5), ra_span * 2 + 1, dec_span * 2 + 1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1 = fig1.add_subplot(122, projection=intensity_list_combined[i * 2 + 1].wcs)
        im1 = ax1.imshow(intensity_list_combined[i * 2 + 1].hdu.data, cmap='cividis')
        ax1.set_xlabel("Right Ascension [hh:mm:ss]")
        ax1.set_ylabel("Declination [deg]", labelpad=-2)
        ax1.text(0.02, 0.98, figure_label[1], transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='top', color='w')
        cbar = plt.colorbar(im1, orientation='horizontal')
        cbar.set_label("Surface Brightness [MJy/sr]", fontsize=axisfontsize)
        rect = patches.Rectangle((-0.5, -0.5), ra_span * 2 + 1, dec_span * 2 + 1,
                                 linewidth=10, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

        #####################################################
        # Plot 2, 4: Extracted spectrum with curve fit and normalised spectrum
        # Curve fit without peaks
        mask = np.full(wave_list[i].shape, True)
        for index in max_inde[i]:
            if index < peak_pixel:
                mask[index:index + peak_pixel * 2 + 1] = False
            else:
                mask[index - peak_pixel:index + peak_pixel + 1] = False
        new_wave = wave_list[i][mask]
        new_spec = spec_list[i][mask]
        fitted_poly = pixel_linfit(pixel_polymodel, new_wave, new_spec)

        fig2 = plt.figure(figsize=figure_size, constrained_layout=True)
        ax1 = fig2.add_subplot(211)
        im11 = ax1.plot(wave_list[i], spec_list[i], label='Extracted Spectrum')
        im12 = ax1.plot(new_wave, fitted_poly(new_wave), color='r', linewidth=1.7, alpha=0.7, label='Curve Fit Model')
        ax1.set_ylabel('Surface Brightness [MJy/sr]', fontsize=axisfontsize)
        ax1.text(0.02, 0.98, figure_label[0], transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax1.legend(loc='upper right')
        ax2 = fig2.add_subplot(212)
        im21 = ax2.plot(wave_resi_list[i], spec_resi_list[i], label='Residual Spectrum')
        im23 = ax2.axhline(0, color='r', linewidth=1.7, alpha=0.7, label='Curve Fit Model')
        for j in max_inde[i]:
            im22 = ax2.axvline(wave_list[i][j] / u.micrometer, linewidth=1, alpha=0.6)
            plt.axvspan((wave_list[i][j] - peak_width / 2) / u.micrometer,
                        (wave_list[i][j] + peak_width / 2) / u.micrometer, alpha=0.1)
        ax2.set_xlabel('Wavelength [µm]', fontsize=axisfontsize)
        ax2.set_ylabel('Surface Brightness [MJy/sr]', fontsize=axisfontsize)
        ax2.text(0.02, 0.98, figure_label[1], transform=ax2.transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax2.legend(loc='upper right')

        #####################################################
        # Plot 5: Extracted normalised spectrum compared to reference
        # Get reference data

        if i > 0:
            ref_name = r"\figure1_digitalized.csv"
            ref_wave = []
            ref_spec = []
            factor = 0.3
            with open(directory + ref_name) as file:
                csvreader = csv.reader(file, delimiter=';')
                for row in csvreader:
                    ref_wave.append(float(row[0]))
                    ref_spec.append(float(row[1]))
            # Convert to mJy
            pixel_size_sr = SCI_data.header['PIXAR_SR']
            compare_spec = spec_resi_list[i] * pixel_size_sr * 1e9 * (2 * ra_span + 1) * (2 * dec_span + 1)
            if filename == r"jw01250006001_03103_00001_nrs2_s3d.fits":
                ax = fig5.add_subplot(211)
                im51 = ax.plot(wave_resi_list[i], compare_spec, label='Extracted Spectrum')
                im52 = ax.plot(ref_wave, ref_spec, label='Reference Spectrum')
                ax.set_xlabel('Wavelength [µm]', fontsize=axisfontsize)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.set_ylabel('Spectral Irradiance  [mJy]', fontsize=axisfontsize)
                ax.text(0.02, 0.72, figure_label[0], transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='top')
                ax.legend(loc='best')
            else:
                ax = fig5.add_subplot(212)
                im51 = ax.plot(wave_resi_list[i], compare_spec, label='Extracted Spectrum')
                im52 = ax.plot(ref_wave, ref_spec, label='Reference Spectrum')
                ax.set_xlabel('Wavelength [µm]', fontsize=axisfontsize)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.set_ylabel('Spectral Irradiance  [mJy]', fontsize=axisfontsize)
                ax.text(0.02, 0.72, figure_label[1], transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='top')
                ax.legend(loc='best')

            save_area = '_Area'
            res = '_Res'

        output_filename1 = ['Spectral_Maps{:s}'.format(res) + '_E{:s}'.format(hdul[0].header['EXPOSURE']) + '.png',
                            'Spectra{:s}'.format(save_area) + '_E{:s}'.format(hdul[0].header['EXPOSURE']) + '.png',
                            'Spectrum_Compare_E{:s}'.format(hdul[0].header['EXPOSURE']) + '.png']
        output_directory = r"C:\Users\steph\Dropbox\Apps\Overleaf\Weissenboeck_thesis\figures\04_Code\Results"
        fig1.savefig(output_directory + r"/" + output_filename1[0], format="png")
        fig2.savefig(output_directory + r"/" + output_filename1[1], format="png")
        if save_area == '_Area':
            fig5.savefig(output_directory + r"/" + output_filename1[2], format="png")

    #####################################################
    # Plot 6, 7, 8, 9: Average Spectral Images around Maximum Peaks
    output_filename2 = ['Spectral_Maps_Peak_E{:s}'.format(hdul[0].header['EXPOSURE'] + '.png'),
                        'Spectral_Maps_Peak_Area_E{:s}'.format(hdul[0].header['EXPOSURE'] + '.png'),
                        'Spectral_Maps_Res_Peak_E{:s}'.format(hdul[0].header['EXPOSURE'] + '.png'),
                        'Spectral_Maps_Res_Peak_Area_E{:s}'.format(hdul[0].header['EXPOSURE'] + '.png')]
    for m in range(0, len(cube_list_combined)):
        fig6789, axs = plt.subplots(3, 3, constrained_layout=True, figsize=figure_size)
        k = 0
        h2o_cube = []
        first_time_k = True
        for l in max_wave[m % 2]:
            spectral_slab_peak = cube_list_combined[m].spectral_slab(l - peak_width / 2, l + peak_width / 2)
            intensity_average_peak = spectral_slab_peak.sum(axis=0) / len(spectral_slab_peak)
            intensity_average_peak[np.isnan(intensity_average_peak)] = 0  # Converts nan values to 0 in intensity
            h2o_cube.append(intensity_average_peak)

        for i in range(0, 3):
            for j in range(0, 3):
                if first_time_k:
                    axs[i, j].text(0.02, 0.98, figure_label[k], transform=axs[i, j].transAxes,
                                   fontsize=16, fontweight='bold', va='top', color='w')
                else:
                    axs[i, j].text(0.02, 0.98, figure_label[k+1], transform=axs[i, j].transAxes,
                                   fontsize=16, fontweight='bold', va='top', color='w')
                if k == 4 and first_time_k:
                    h2o_cube_sum = sum(h2o_cube) / len(max_inde[m % 2])
                    im = axs[i, j].imshow(h2o_cube_sum.hdu.data, cmap='cividis', norm=SymLogNorm(linthresh=1))
                    axs[i, j].set_title("All Peaks", fontsize=axisfontsize)
                    axs[i, j].xaxis.set_visible(False)
                    axs[i, j].yaxis.set_visible(False)
                    axs[i, j].invert_yaxis()
                    first_time_k = False
                    cbar = plt.colorbar(im)
                    cbar.set_label("Sur. Bright. [MJy/sr]", fontsize=axisfontsize)
                else:
                    im = axs[i, j].imshow(h2o_cube[k].hdu.data, cmap='cividis', norm=SymLogNorm(linthresh=1))
                    axs[i, j].set_title("Peak at {:.3f}".format(max_wave[m % 2][k]), fontsize=axisfontsize)
                    axs[i, j].xaxis.set_visible(False)
                    axs[i, j].yaxis.set_visible(False)
                    axs[i, j].invert_yaxis()
                    cbar = plt.colorbar(im)
                    cbar.set_label("Sur. Bright. [MJy/sr]", fontsize=axisfontsize)
                    k += 1

        fig6789.savefig(output_directory + r"/" + output_filename2[m], format="png")

    # endregion 4)
plt.show()