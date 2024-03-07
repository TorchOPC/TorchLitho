'''

This code aim to calculate the 1D aerial image
sr: source
mk: mask
po: projector
rp: receipe
numerics: numerics

'''

import torch
import sys, os

dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/../..")

from CalculateCharacteristicMatrix import CalculateCharacteristicMatrix
from litho.Numerics import Numerics
from litho.Source import Source
from litho.Receipe import Receipe
from litho.Mask import Mask
from litho.ProjectionObjective import ProjectionObjective
from litho.ImageData import ImageData

def cartesian_to_polar(x, y):
    
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    return rho, theta

def Calculate1DAerialImage(sr, mk, po,aerail_litho_image, rp, numerics):
    
    mask_nf = numerics.SampleNumber_Mask_X  # the sample number of mask
    wafer_nf = numerics.SampleNumber_Wafer_X # the sample number of wafer

    # Get the number of sample souurce
    sr.PntNum = numerics.SampleNumber_Source
    sourceData = sr.Calc_SourceSimple()

    # Calculate the weight of source
    weight = sourceData.Value.sum()
    wavelength = sr.Wavelength
    NA = po.NA                  # Numerical aperture
    M = po.Reduction            # Zoom ratio
    
    # Select the index of refraction according to the lens type
    if po.LensType == 'Immersion':
        indexImage = po.Index_ImmersionLiquid
    elif po.LensType == 'Dry':
        indexImage = 1
        if NA >= 1:
            raise ValueError("Wrong NA!")
    else:
        raise ValueError("Unsupported Lens Type!")

    # Check mask rotation
    if mk.Orientation == 0:
        pass
    elif torch.abs(mk.Orientation - torch.pi / 2) < torch.finfo(float).eps:
        tenpY = sourceData.Y
        sourceData.Y = -sourceData.X
        sourceData.X = tenpY
    else:
        raise ValueError('Not supported orientation angle')

    # Calculate mask spectrum
    mk.Nf = mask_nf
    spectrum, mask_fs, *_ = mk.CalculateMaskSpectrum(po, sr)

    # Calculate the range of simulation
    simulation_range = rp.FocusRange - rp.Focus
    # Initialize the intensity matrix
    intensity = torch.zeros(simulation_range, wafer_nf - 1)

    exyz_calculate_number = 1

    
    # Extended coordinate information for vectorized calculations
    fm_s = torch.t(sourceData.X.repeat(mask_nf - 1, 1))
    gm_s = torch.t(sourceData.Y.repeat(mask_nf - 1, 1))

    # Calculate frequency-domain coordinates of mask sample points
    fm_sm = fm_s + mask_fs[:-1]
    gm_sm = gm_s

    rho2 = fm_sm ** 2 + gm_sm ** 2
    
    # Effective pore size limited by normalized pore size
    valid_pupil = (rho2 <= 1)

    fm_sm = torch.t(fm_sm)
    gm_sm = torch.t(gm_sm)

    valid_pupil = torch.t(valid_pupil)
    f_calc = fm_sm[valid_pupil]
    g_calc = gm_sm[valid_pupil]

    # Convert Cartesian Coordinates to Polar Coordinates
    rho_calc, theta_calc = cartesian_to_polar(f_calc, g_calc)
    fg_square = rho_calc ** 2

    # Compute the tilt factor
    obliquity_factor = torch.sqrt(torch.sqrt((1 - (M ** 2 * NA ** 2) * fg_square) /
                                (1 - (NA / indexImage) ** 2 * fg_square)))

    # Calculating Waveform Aberrations
    if mk.Orientation == 0:
        aberration = po.CalculateAberrationFast(rho_calc, theta_calc, 0)
    elif torch.abs(mk.Orientation - torch.pi / 2) < torch.finfo(float).eps:
        aberration = po.CalculateAberrationFast(rho_calc, theta_calc, torch.pi / 2)
    else:
        raise ValueError('Not supported orientation angle')

    siz = spectrum.size(0)
    spectrum_calc = spectrum.view(-1)[:siz*siz-1]*torch.ones(sourceData.Value.size(0),1)

    spectrum_calc = torch.t(spectrum_calc[:,:valid_pupil.size(0)])
    spectrum_calc = spectrum_calc[valid_pupil]
    temp_h0_aber = spectrum_calc * obliquity_factor * torch.exp(1j * 2 * torch.pi * aberration)

    # Initialize the tilted ray matrix
    oblique_rays_matrix = torch.ones(len(rho_calc), exyz_calculate_number)
    
    if numerics.ImageCalculationMode == 'vector':
        exyz_calculate_number = 3
        
        fm_s = torch.t(fm_s)
        gm_s = torch.t(gm_s)
        f_calc_s = fm_s[valid_pupil]
        g_calc_s = gm_s[valid_pupil]
        nrs, nts = cartesian_to_polar(f_calc_s, g_calc_s)

        # polarization decomposition
        polarized_x, polarized_y = sr.Calc_PolarizationMap(nts, nrs)
        # Calculate the polarization transfer matrix

        m0xx, m0yx, m0xy, m0yy, m0xz, m0yz = CalculateCharacteristicMatrix(f_calc, g_calc, fg_square, NA, indexImage)

        oblique_rays_matrix = torch.ones(len(rho_calc), exyz_calculate_number)
        oblique_rays_matrix[:, 0] = polarized_x * m0xx + polarized_y * m0yx
        oblique_rays_matrix[:, 1] = polarized_x * m0xy + polarized_y * m0yy
        oblique_rays_matrix[:, 2] = polarized_x * m0xz + polarized_y * m0yz

    # whether pupil function exists
    if po.PupilFilter['Type'] != 'none':
        filter_fn = getattr(po.PupilFilter, po.PupilFilter.Type)
        parameter = po.PupilFilter.Parameter
        f_pupil = f_calc
        g_pupil = g_calc
        pupil_filter_data = filter_fn(parameter, f_pupil, g_pupil)
        temp_h0_aber = pupil_filter_data * temp_h0_aber
        
    # Calculate defocus
    temp_focus = -1j * 2 * torch.pi / wavelength * torch.sqrt(indexImage ** 2 - NA ** 2 * fg_square)

    valid_pupil = torch.t(valid_pupil)
    
    # Calculate the intensity distribution
    for i_focus in range(len(simulation_range)):
        intensity = torch.zeros(1, wafer_nf - 1)
        temp_f = torch.exp(temp_focus * simulation_range[i_focus])
        for i_em in range(exyz_calculate_number):
            rho2[:] = 0
            rho2 = torch.t(rho2)
            valid_pupil = torch.t(valid_pupil)
            rho2[valid_pupil] = oblique_rays_matrix[:, i_em] * torch.real(temp_h0_aber) * torch.real(temp_f)
            rho2 = torch.t(rho2)
            valid_pupil = torch.t(valid_pupil)
            # Interpolate the frequency domain signal to match the number of wafer sample points
            if wafer_nf == mask_nf:
                exyz_frequency = rho2
            elif wafer_nf > mask_nf:
                padding = (wafer_nf - mask_nf) // 2
                exyz_frequency = torch.cat((torch.zeros(len(sourceData.Value), padding), rho2,
                                            torch.zeros(len(sourceData.Value), padding)), dim=1)
            else:
                exyz_frequency = rho2
                exyz_frequency[:, : (mask_nf - wafer_nf) // 2] = torch.tensor([])
                exyz_frequency[:, -((mask_nf - wafer_nf) // 2):] = torch.tensor([])

            exyz = torch.fft.fft(exyz_frequency, dim=1)

            intensity_con = torch.real(exyz) ** 2 + torch.imag(exyz) ** 2

            # Calculate and accumulate intensity
            intensity_temp = indexImage * (mask_fs[1] - mask_fs[0]) ** 2 * torch.fft.fftshift(
                sourceData.Value.unsqueeze(0) @ intensity_con)

            intensity += intensity_temp
        # normalized intensity
        intensity[i_focus, :] = intensity / weight
    
    image_x = torch.linspace(-mk.Period_X / 2, mk.Period_X / 2, wafer_nf)
    image_y = torch.tensor(0)
    image_z = rp.FocusRange
    aerail_litho_image.ImageType = '1d'
    aerail_litho_image.SimulationType = 'aerial'
    aerail_litho_image.Intensity = torch.flip(torch.cat((intensity, intensity[:, 0].unsqueeze(1)), dim=1), dims=[1])
    aerail_litho_image.ImageX = image_x
    aerail_litho_image.ImageY = image_y
    aerail_litho_image.ImageZ = image_z

    return aerail_litho_image


# Define a function to check the correctness of Calculate1DAerialImage
def check():
    sr = Source()
    mk = Mask()  # Initialize with appropriate values
    mk = mk.CreateLineMask(45, 90)
    po = ProjectionObjective()  # Initialize with appropriate values
    rp = Receipe()  # Initialize with appropriate values
    numerics = Numerics()  # Initialize with appropriate values
    aerail_litho_image = ImageData()

    # Call the function to be tested
    result = Calculate1DAerialImage(sr, mk, po, aerail_litho_image, rp, numerics)
    
    # Print some validation information (you can add more checks)
    print("Intensity:", result.Intensity)
    print("Intensity sum:", torch.sum(result.Intensity))
    print("ImageX:", result.ImageX)
    print("ImageY:", result.ImageY)
    print("ImageZ:", result.ImageZ)

if __name__ == '__main__':
    # Call the check function to test Calculate1DAerialImage
    check()