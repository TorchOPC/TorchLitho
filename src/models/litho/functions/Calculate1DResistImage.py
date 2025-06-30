'''
This code aim to calculate the 1D resist image
'''

import torch
import sys, os

dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/../..")

from CalculateTransferMatrix import CalculateTransferMatrix
from litho.Numerics import Numerics
from litho.Source import Source
from litho.Receipe import Receipe
from litho.Mask import Mask
from litho.ProjectionObjective import ProjectionObjective
from litho.ImageData import ImageData
from litho.FilmStack import FilmStack

def cartesian_to_polar(x, y):
    
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    return rho, theta


def Calculate1DResistImage(source, mask, projector, filmStack, lithoImage, receipe, numerics):
    mask_nf = numerics.SampleNumber_Mask_X
    waferNf = numerics.SampleNumber_Wafer_X

    source.PntNum = numerics.SampleNumber_Source
    sourceData = source.Calc_SourceSimple()
    weightSource = torch.sum(sourceData.Value)
    wavelength = source.Wavelength
    NA = projector.NA
    M = projector.Reduction

    if projector.LensType == 'Immersion':
        indexImage = projector.Index_ImmersionLiquid
    elif projector.LensType == 'Dry':
        indexImage = 1
        if NA >= 1:
            raise ValueError('Wrong NA!')
    else:
        raise ValueError('Unsupported Lens Type')

    if mask.Orientation == 0:
        pass
    elif torch.abs(mask.Orientation - torch.pi/2) < torch.finfo(torch.float32).eps:
        tenpY = sourceData.Y
        sourceData.Y = -sourceData.X
        sourceData.X = tenpY
    else:
        raise ValueError('Not supported orientation angle')

    mask.Nf = mask_nf
    spectrum, mask_fs, *_ = mask.CalculateMaskSpectrum(projector, source)

    if numerics.ImageCalculationMode.lower() != 'vector':
        raise ValueError('Projector model must be vector')


    if receipe.Focus.dim() == 0:
        pass
    if receipe.Focus.dim() == 1:
        if len(receipe.Focus) > 1:
            raise ValueError('Not support multi focus')

    # Extended coordinate information for vectorized calculations
    fm_s = torch.t(sourceData.X.repeat(mask_nf - 1, 1))
    gm_s = torch.t(sourceData.Y.repeat(mask_nf - 1, 1))
    
    # Calculate frequency-domain coordinates of mask sample points
    fm_sm = fm_s + mask_fs[:-1]
    gm_sm = gm_s

    rho2 = fm_sm ** 2 + gm_sm ** 2
    valid_pupil = (rho2 <= 1)
    
    fm_sm = torch.t(fm_sm)
    gm_sm = torch.t(gm_sm)
    valid_pupil = torch.t(valid_pupil)
    
    f_calc = fm_sm[valid_pupil]
    g_calc = gm_sm[valid_pupil]
    
    rho_calc, theta_calc = cartesian_to_polar(f_calc, g_calc)
    fgSquare = rho_calc ** 2

    obliquity_factor = torch.sqrt(torch.sqrt((1 - (M ** 2 * NA ** 2) * 
                    fgSquare) / (1 - (NA / indexImage) ** 2 * fgSquare)))

    if mask.Orientation == 0:
        aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, 0)
    elif torch.abs(mask.Orientation - torch.pi/2) < torch.finfo(torch.float32).eps:
        aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, torch.pi/2)
    else:
        raise ValueError('Not supported orientation angle')

    siz = spectrum.size(0)
    spectrum_calc = spectrum.view(-1)[:siz*siz-1]*torch.ones(sourceData.Value.size(0),1)

    spectrum_calc = torch.t(spectrum_calc[:,:valid_pupil.size(0)])
    spectrum_calc = spectrum_calc[valid_pupil]
    temp_h0_aber = spectrum_calc * obliquity_factor * torch.exp(1j * 2 * torch.pi * aberration)
    
    fm_s = torch.t(fm_s)
    gm_s = torch.t(gm_s)
    f_calc_s = fm_s[valid_pupil]
    g_calc_s = gm_s[valid_pupil]
    nrs, nts = cartesian_to_polar(f_calc_s, g_calc_s)
    PolarizedX, PolarizedY = source.Calc_PolarizationMap(nts, nrs)

    alpha = (NA / indexImage) * f_calc
    beta = (NA / indexImage) * g_calc
    gamma = torch.sqrt(1 - (NA / indexImage) ** 2 * fgSquare)
    vectorRho2 = 1 - gamma ** 2

    Pxsx = (beta ** 2) / vectorRho2
    Pysx = (-alpha * beta) / vectorRho2
    Pxsy = Pysx
    Pysy = (alpha ** 2) / vectorRho2
    Pxpx = gamma * Pysy
    Pypx = -gamma * Pysx
    Pxpy = Pypx
    Pypy = gamma * Pxsx
    Pxpz = -alpha
    Pypz = -beta

    centerpoint = torch.nonzero(fgSquare < torch.finfo(torch.float32).eps).squeeze()
    if centerpoint.numel() > torch.finfo(torch.float32).eps:
        Pxsx[centerpoint] = 0
        Pysx[centerpoint] = 0
        Pxsy[centerpoint] = 0
        Pysy[centerpoint] = 0
        Pxpx[centerpoint] = 0
        Pypx[centerpoint] = 0
        Pxpy[centerpoint] = 0
        Pypy[centerpoint] = 0
        Pxpz[centerpoint] = 0
        Pypz[centerpoint] = 0

    if projector.PupilFilter['Type'] != 'none':
        filter_fn = getattr(projector.PupilFilter, projector.PupilFilter.Type)
        parameter = projector.PupilFilter.Parameter
        f_pupil = f_calc
        g_pupil = g_calc
        pupil_filter_data = filter_fn(parameter, f_pupil, g_pupil)
        temp_h0_aber = pupil_filter_data * temp_h0_aber


    thcknessResist = filmStack.GetResistThickness()
    indexResist = filmStack.GetResistIndex()
    indexSubstrate = filmStack.GetSubstrateIndex()
    TARCs = filmStack.GetTARCLayers()
    BARCs = filmStack.GetBARCLayers()
    eta0 = torch.sqrt(Numerics.Mu0 / Numerics.Epsilon0)

    cosThetaInc = gamma
    sinThetaInc = torch.sqrt(1 - gamma**2)

    sinThetaResist = sinThetaInc * indexImage / indexResist
    cosThetaResist = torch.sqrt(1 - sinThetaResist**2)
    sinThetaSubstrate = sinThetaInc * indexImage / indexSubstrate
    cosThetaSubstrate = torch.sqrt(1 - sinThetaSubstrate**2)

    etaInc = eta0 / indexImage
    etaResist = eta0 / indexResist
    etaSubstrate = eta0 / indexSubstrate

    faiInc = cosThetaInc / etaInc
    faiResist = cosThetaResist / etaResist
    M11, M12, M21, M22 = CalculateTransferMatrix(TARCs, sinThetaInc, indexImage, wavelength, 'TE')
    rhoStacks = (faiInc * (M11 - faiResist * M12) + (M21 - faiResist * M22)) / (faiInc * (M11 - faiResist * M12) - (M21 - faiResist * M22))
    tauStacks = 2 * faiInc / (faiInc * (M11 - faiResist * M12) - (M21 - faiResist * M22))

    # BottomARCs
    faiSubstrate = cosThetaSubstrate / etaSubstrate
    M11, M12, M21, M22 = CalculateTransferMatrix(BARCs, sinThetaInc, indexImage, wavelength, 'TE')
    rhoSubStacks = (faiResist * (M11 - faiSubstrate * M12) + (M21 - faiSubstrate * M22)) / (faiResist * (M11 - faiSubstrate * M12) - (M21 - faiSubstrate * M22))

    # TopARCs
    faiIncTM = -cosThetaInc * etaInc
    faiResistTM = -cosThetaResist * etaResist
    M11, M12, M21, M22 = CalculateTransferMatrix(TARCs, sinThetaInc, indexImage, wavelength, 'TM')
    rhoStackp = (faiIncTM * (M11 - faiResistTM * M12) + (M21 - faiResistTM * M22)) / (faiIncTM * (M11 - faiResistTM * M12) - (M21 - faiResistTM * M22))
    tauStackp = -2 * etaResist * cosThetaInc / (faiIncTM * (M11 - faiResistTM * M12) - (M21 - faiResistTM * M22))

    # BottomARCs
    faiSubstrateTM = -cosThetaSubstrate * etaSubstrate
    M11, M12, M21, M22 = CalculateTransferMatrix(BARCs, sinThetaInc, indexImage, wavelength, 'TM')
    rhoSubStackp = (faiResistTM * (M11 - faiSubstrateTM * M12) + (M21 - faiSubstrateTM * M22)) / (faiResistTM * (M11 - faiSubstrateTM * M12) - (M21 - faiSubstrateTM * M22))


    tempF = torch.exp(-1j * 2 * torch.pi / wavelength * torch.sqrt(indexImage**2 - NA*NA*fgSquare) * (-receipe.Focus))
    TempHAber = temp_h0_aber * tempF

    if numerics.SimulationRange_Resist:
        if numerics.SimulationRange_Resist[0] < 0 or numerics.SimulationRange_Resist[-1] > thcknessResist:
            raise ValueError('Wrong focus range')
        Sample_RIZ = len(numerics.SimulationRange_Resist)
        ImageZ_Resist = numerics.SimulationRange_Resist
    else:
        Sample_RIZ = numerics.SampleNumber_Wafer_Z
        ImageZ_Resist = torch.linspace(0, thcknessResist, Sample_RIZ)

    IntensityResist = torch.zeros((Sample_RIZ, mask_nf - 1), dtype=torch.complex128)
    expWaveD = torch.exp(2 * 1j * 2 * torch.pi / wavelength * indexResist * cosThetaResist * thcknessResist)
    rho2 = torch.zeros(valid_pupil.shape, dtype=torch.complex128)

    for idepth in range(0, Sample_RIZ):
        waveZ = 1j * 2 * torch.pi / wavelength * indexResist * cosThetaResist * (thcknessResist - ImageZ_Resist[idepth])
        expPosWaveZ = torch.exp(waveZ)
        expNegWaveDZ = expWaveD / expPosWaveZ

        Fs = tauStacks / (1 + rhoStacks * rhoSubStacks * expWaveD) * (expPosWaveZ + rhoSubStacks * expNegWaveDZ)
        Fpxy = tauStackp / (1 + rhoStackp * rhoSubStackp * expWaveD) * (expPosWaveZ - rhoSubStackp * expNegWaveDZ)
        Fpz = tauStackp / (1 + rhoStackp * rhoSubStackp * expWaveD) * (expPosWaveZ + rhoSubStackp * expNegWaveDZ)

        MSxx = Fs * Pxsx + Fpxy * Pxpx
        MSyx = Fs * Pysx + Fpxy * Pypx
        MSxy = Fs * Pxsy + Fpxy * Pxpy
        MSyy = Fs * Pysy + Fpxy * Pypy
        MSxz = Fpz * Pxpz
        MSyz = Fpz * Pypz
        
        obliqueRaysMatrixX = PolarizedX * MSxx + PolarizedY * MSyx
        obliqueRaysMatrixY = PolarizedX * MSxy + PolarizedY * MSyy
        obliqueRaysMatrixZ = PolarizedX * MSxz + PolarizedY * MSyz
        
        rho2[valid_pupil] = (obliqueRaysMatrixX * TempHAber).to(torch.complex128)
        Ex = torch.fft.fft(rho2, dim=0)
        rho2[valid_pupil] = (obliqueRaysMatrixY * TempHAber).to(torch.complex128)
        Ey = torch.fft.fft(rho2, dim=0)
        rho2[valid_pupil] = (obliqueRaysMatrixZ * TempHAber).to(torch.complex128)
        Ez = torch.fft.fft(rho2, dim=0)

        IntensityCon = torch.real(Ex)**2 + torch.imag(Ex)**2 + torch.real(Ey)**2 + torch.imag(Ey)**2 + torch.real(Ez)**2 + torch.imag(Ez)**2
        IntensityCon = IntensityCon.float()

        IntensityTemp = indexResist.real * (mask_fs[1] - mask_fs[0])**2 / weightSource * torch.fft.fftshift(torch.matmul(sourceData.Value.T, IntensityCon.T),dim=0)

        IntensityResist[idepth, :] = IntensityTemp

    ImageX = torch.linspace(-mask.Period_X/2, mask.Period_X/2, waferNf)
    ImageY = torch.tensor(0)
    ImageZ = ImageZ_Resist

    calcNf = mask_nf

    if waferNf == calcNf:
        intensityOutPut = torch.cat((IntensityResist, IntensityResist[:, 0].unsqueeze(1)), dim=1)
    elif waferNf > calcNf:
        IntensityFrequency = torch.fft.fftshift(torch.fft.fft(IntensityResist, dim=1), dims=1)
        IntensityFrequency = torch.cat((IntensityFrequency, IntensityResist[:, 0].unsqueeze(1)), dim=1)
        pad_width = (0, (waferNf - calcNf) // 2)
        IntensityFrequencyWafer = torch.nn.functional.pad(IntensityFrequency, (0, 0, pad_width[0], pad_width[1]), 'constant', 0)
        IntensityFrequencyWafer = torch.fft.fftshift(IntensityFrequencyWafer, dims=1)
        IntensityWafer = torch.abs(torch.fft.ifft(IntensityFrequencyWafer, dim=1)) * (waferNf-1)/(calcNf-1)
        intensityOutPut = torch.cat((IntensityWafer, IntensityResist[:, 0].unsqueeze(1)), dim=1)
    else:
        raise ValueError('WaferNf must be more than calcNf!')

    lithoImage.ImageType = '1d'
    lithoImage.SimulationType = 'resist'
    lithoImage.Intensity = intensityOutPut
    lithoImage.ImageX = ImageX
    lithoImage.ImageY = ImageY
    lithoImage.ImageZ = ImageZ

    return lithoImage

# Define a function to check the correctness of Calculate1DResistImage
def check():
    sr = Source()
    mk = Mask() 
    mk.CreateLineMask(45, 90) 
    po = ProjectionObjective()  
    filmStack = FilmStack()
    rp = Receipe()  
    numerics = Numerics()  
    lithoImage = ImageData()

    # Call the function to be tested
    result = Calculate1DResistImage(sr, mk, po, filmStack, lithoImage, rp, numerics)
    
    
    # Print some validation information (you can add more checks)
    print("Intensity shape:", result.Intensity.shape)
    print("Intensity:", result.Intensity)
    print("Intensity:", torch.sum(result.Intensity))
    print("ImageX:", result.ImageX)
    print("ImageY:", result.ImageY)
    print("ImageZ:", result.ImageZ)

if __name__ == '__main__':
    # Call the check function to test Calculate1DAerialImage
    check()