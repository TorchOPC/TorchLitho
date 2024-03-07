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

def Calculate2DResistImage(source, mask, projector, filmStack, resistLithoImage, receipe, numerics):

    maskNf = numerics.SampleNumber_Mask_X
    maskNg = numerics.SampleNumber_Mask_Y
    waferNf = numerics.SampleNumber_Wafer_X
    waferNg = numerics.SampleNumber_Wafer_Y
    waferNz = numerics.SampleNumber_Wafer_Z

    source.PntNum = numerics.SampleNumber_Source
    sourceData = source.Calc_SourceSimple()

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


    mask.Nf = maskNf
    mask.Ng = maskNg
    spectrumMask, mask_fs, mask_gs, _ = mask.CalculateMaskSpectrum(projector, source)

    index_calc_fs = torch.abs(mask_fs) < 2
    index_calc_gs = torch.abs(mask_gs) < 2
    mask_calc_fs = mask_fs[index_calc_fs]
    mask_calc_gs = mask_gs[index_calc_gs]
    spectrumCalc = spectrumMask[index_calc_gs][:, index_calc_fs]
    calcNf = len(mask_calc_fs)
    calcNg = len(mask_calc_gs)

    Orientation = mask.Orientation

    if numerics.ImageCalculationMode != 'vector':
        raise ValueError('Projector model must be vector')

    if receipe.Focus.dim() == 0:
        pass
    if receipe.Focus.dim() == 1:
        if len(receipe.Focus) > 1:
            raise ValueError('Not support multi focus')

    mask_fm, mask_gm = torch.meshgrid(mask_calc_fs[:-1], mask_calc_gs[:-1])

    sourceX = sourceData.X
    sourceY = sourceData.Y
    sourceV = sourceData.Value
    weight = sourceV.sum()

    source_rho, source_theta = cartesian_to_polar(sourceX, sourceY)
    PolarizedX, PolarizedY = source.Calc_PolarizationMap(source_theta, source_rho)
    new_spectrum = spectrumCalc[:-1, :-1]
    mask_fg2m = mask_fm**2 + mask_gm**2
    sourceXY2 = sourceX**2 + sourceY**2

    thcknessResist = filmStack.GetResistThickness()
    indexResist = filmStack.GetResistIndex()
    indexSubstrate = filmStack.GetSubstrateIndex()
    TARCs = filmStack.GetTARCLayers()
    BARCs = filmStack.GetBARCLayers()
    eta0 = torch.sqrt(Numerics.Mu0 / Numerics.Epsilon0)

    if len(numerics.SimulationRange_Resist) == 0:
        ImageZ_Resist = torch.linspace(0, thcknessResist, waferNz)
    else:
        ImageZ_Resist = numerics.SimulationRange_Resist

    # Exyzp = torch.zeros(calcNg-1, calcNf-1, len(ImageZ_Resist), dtype=torch.complex64)
    # intensity3D = torch.zeros(calcNg-1, calcNf-1, len(ImageZ_Resist))
    Exyzp = torch.zeros(len(ImageZ_Resist), calcNf-1, calcNg-1, dtype=torch.complex64)
    intensity3D = torch.zeros(len(ImageZ_Resist), calcNf-1, calcNg-1)
    for iSource in range(len(sourceV)):
        rho2 = mask_fg2m + 2*(sourceX[iSource]*mask_fm + sourceY[iSource]*mask_gm) + sourceXY2[iSource]
        
        validPupil = (rho2 <= 1)
        f_calc = mask_fm[validPupil] + sourceX[iSource]
        g_calc = mask_gm[validPupil] + sourceY[iSource]
        rho_calc, theta_calc = cartesian_to_polar(f_calc, g_calc)
        fgSquare = rho_calc**2
        alpha = (NA/indexImage) * f_calc
        beta = (NA/indexImage) * g_calc
        gamma = torch.sqrt(1 - (NA/indexImage)**2 * fgSquare)
        
        obliquityFactor = torch.sqrt(torch.sqrt((1 - (M**2*NA**2) * fgSquare) / (1 - ((NA/indexImage)**2) * fgSquare)))  # 倾斜因子
        aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, Orientation)  # 波像差
        pupilFilter = projector.CalculatePupilFilter(rho_calc, theta_calc)  # 光瞳函数
        tempFocus = torch.exp(-1j*2*torch.pi/wavelength * torch.sqrt(indexImage**2 - NA**2 * fgSquare) * -receipe.Focus)  # 离焦
        SpectrumCalc = new_spectrum[validPupil]
        TempHAber = SpectrumCalc * obliquityFactor * torch.exp(1j*2*torch.pi*aberration) * pupilFilter * tempFocus  # 传递函数

        vectorRho2 = alpha**2 + beta**2
        Pxsx = (beta**2) / vectorRho2
        Pysx = (-alpha * beta) / vectorRho2
        Pxsy = Pysx
        Pysy = (alpha**2) / vectorRho2

        Pxpx = gamma * Pysy
        Pypx = -gamma * Pysx
        Pxpy = Pypx
        Pypy = gamma * Pxsx
        Pxpz = -alpha
        Pypz = -beta

        centerpoint = torch.nonzero(vectorRho2 < torch.finfo(torch.float32).eps)
        if len(centerpoint) > torch.finfo(torch.float32).eps:
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

        cosThetaInc = gamma
        sinThetaInc = torch.sqrt(1 - gamma**2)
        sinThetaResist = sinThetaInc * indexImage / indexResist
        cosThetaResist = torch.sqrt(1 - sinThetaResist**2)
        sinThetaSubstrate = sinThetaInc * indexImage / indexSubstrate
        cosThetaSubstrate = torch.sqrt(1 - sinThetaSubstrate**2)

        etaInc = eta0 / indexImage
        etaResist = eta0 / indexResist
        etaSubstrate = eta0 / indexSubstrate


        # TopARCs
        faiInc = cosThetaInc / etaInc
        faiResist = cosThetaResist / etaResist
        M11, M12, M21, M22 = CalculateTransferMatrix(TARCs, sinThetaInc, indexImage, wavelength, 'TE')
        rhoStacks = (faiInc * (M11 - faiResist * M12) + (M21 - faiResist * M22)) / (faiInc * (M11 - faiResist * M12) - (M21 - faiResist * M22))
        tauStacks = 2 * faiInc / (faiInc * (M11 - faiResist * M12) - (M21 - faiResist * M22))

        # BottomArcs
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

        expWaveD2 = torch.exp(2j * 2 * torch.pi / wavelength * indexResist * cosThetaResist * thcknessResist)
        A2FsH = tauStacks / (1 + rhoStacks * rhoSubStacks * expWaveD2) * TempHAber
        A2FpxyH = tauStackp / (1 + rhoStackp * rhoSubStackp * expWaveD2) * TempHAber
        A2FpzH = A2FpxyH

        temp = torch.full_like(ImageZ_Resist, thcknessResist) - ImageZ_Resist
        temp = temp.view(1, temp.size(0))
        expPosWaveZ = torch.exp(1j * 2 * torch.pi / wavelength * indexResist * cosThetaResist.view(cosThetaResist.size(0),1) * temp)
        expNegWaveD2Z = expWaveD2.view(expWaveD2.size(0), 1) / expPosWaveZ

        rhoSubStacks = rhoSubStacks.view(rhoSubStacks.size(0), 1)
        rhoSubStackp = rhoSubStackp.view(rhoSubStackp.size(0), 1)
        A2FsH = A2FsH.view(A2FsH.size(0), 1)
        A2FpxyH = A2FpxyH.view(A2FpxyH.size(0), 1)
        A2FpzH = A2FpzH.view(A2FpzH.size(0), 1)
        
        Fs = A2FsH* (expPosWaveZ + rhoSubStacks * expNegWaveD2Z)
        Fpxy = A2FpxyH * (expPosWaveZ - rhoSubStackp * expNegWaveD2Z)
        Fpz = A2FpzH * (expPosWaveZ + rhoSubStackp * expNegWaveD2Z)
        Pxsx = Pxsx.view(Pxsx.size(0), 1)
        Pxpx = Pxpx.view(Pxpx.size(0), 1)
        Pysx = Pysx.view(Pysx.size(0), 1)
        Pypx = Pypx.view(Pypx.size(0), 1)
        Pxsy = Pxsy.view(Pxsy.size(0), 1)
        Pxpy = Pxpy.view(Pxpy.size(0), 1)
        Pysy = Pysy.view(Pysy.size(0), 1)
        Pypy = Pypy.view(Pypy.size(0), 1)
        Pxpz = Pxpz.view(Pxpz.size(0), 1)
        Pypz = Pypz.view(Pypz.size(0), 1)
        
        MSxx = Fs * Pxsx + Fpxy * Pxpx
        MSyx = Fs * Pysx + Fpxy * Pypx
        MSxy = Fs * Pxsy + Fpxy * Pxpy
        MSyy = Fs * Pysy + Fpxy * Pypy
        MSxz = Fpz * Pxpz
        MSyz = Fpz * Pypz

        ExValid = PolarizedX[iSource] * MSxx + PolarizedY[iSource] * MSyx
        EyValid = PolarizedX[iSource] * MSxy + PolarizedY[iSource] * MSyy
        EzValid = PolarizedX[iSource] * MSxz + PolarizedY[iSource] * MSyz

        zIndex = torch.arange(0, len(ImageZ_Resist) * (calcNf - 1) * (calcNg - 1), (calcNf - 1) * (calcNg - 1))
        validPupil = torch.nonzero(rho2.flatten() <= 1)
        validPupil = validPupil.view(validPupil.size(0), 1)
        zIndex = zIndex.view(1, zIndex.size(0))
        validPupilIndexFull = (validPupil + zIndex)
        validPupilIndexFull = validPupilIndexFull.view(-1)

        Ex = torch.zeros_like(Exyzp,dtype = torch.complex64)
        Ey = torch.zeros_like(Exyzp,dtype = torch.complex64)
        Ez = torch.zeros_like(Exyzp,dtype = torch.complex64)
        
        a = Ex.size(0)
        b = Ex.size(1)
        c = Ex.size(2)
        
        Ex = Ex.view(-1)
        Ey = Ey.view(-1)
        Ez = Ez.view(-1)
        Ex[validPupilIndexFull] = ExValid.view(-1).to(Ex.dtype)
        Ey[validPupilIndexFull] = EyValid.view(-1).to(Ey.dtype)
        Ez[validPupilIndexFull] = EzValid.view(-1).to(Ez.dtype)

        Ex = Ex.view(a,b,c)
        Ey = Ey.view(a,b,c)
        Ez = Ez.view(a,b,c)

        Ex = torch.fft.fft2(Ex)
        Ey = torch.fft.fft2(Ey)
        Ez = torch.fft.fft2(Ez)
        intensityTemp = torch.real(Ex)**2 + torch.imag(Ex)**2 + torch.real(Ey)**2 + torch.imag(Ey)**2 + torch.real(Ez)**2 + torch.imag(Ez)**2
        intensity3D = intensity3D + intensityTemp * sourceV[iSource]


    dfmdg = (mask_fs[1] - mask_fs[0]) * (mask_gs[1] - mask_gs[0])
    intensity3D = dfmdg**2 * torch.fft.fftshift(torch.fft.fftshift(intensity3D, dim=2), dim=1)

    intensity3D = indexResist.real * intensity3D / weight

    if waferNf == calcNf and waferNg == calcNg:
        intensityOutPut = intensity3D
    elif waferNf < calcNf or waferNg < calcNg:
        raise ValueError('Wafer grid must be larger than calculate grid!')
    else:
        intensityOutPut = torch.zeros(len(ImageZ_Resist), waferNf - 1, waferNg - 1)

        intensity3DFrequency = torch.fft.fftshift(torch.fft.fftshift(torch.fft.fft2(intensity3D), dim=2), dim=1)
        intensity3DFrequency = torch.cat((intensity3DFrequency, intensity3DFrequency[:, 0, :].unsqueeze(1)), dim = 1)
        intensity3DFrequency = torch.cat((intensity3DFrequency, intensity3DFrequency[:, :, 0].unsqueeze(2)), dim = 2)
        rangeNg = torch.arange((waferNg + 1) // 2 - (calcNg - 1) // 2 - 1, (waferNg + 1) // 2 + (calcNg - 1) // 2)
        rangeNf = torch.arange((waferNf + 1) // 2 - (calcNf - 1) // 2 - 1, (waferNf + 1) // 2 + (calcNf - 1) // 2)
        # rangeNg = rangeNg.view(1, -1)
        # rangeNf = rangeNf.view(1, -1)
      
        intensity3DFrequency = intensity3DFrequency.to(intensityOutPut.dtype)
        
        for i in range(len(rangeNg)):
            for j in range(len(rangeNf)):
                intensityOutPut[:,rangeNf[j], rangeNg[i]] = intensity3DFrequency[:, j, i]
            
        intensityOutPut = torch.fft.fftshift(torch.fft.fftshift(intensityOutPut, dim=2), dim=1)
        intensityOutPut = torch.abs(torch.fft.ifft2(intensityOutPut)) * (waferNf - 1) / (calcNf - 1) * (waferNg - 1) / (calcNg - 1)
    intensityOutPut = torch.cat((intensityOutPut, intensityOutPut[:, 0, :].unsqueeze(1)), dim = 1)
    intensityOutPut = torch.cat((intensityOutPut, intensityOutPut[:, :, 0].unsqueeze(2)), dim = 2)
    intensityOutPut = torch.transpose(intensityOutPut, 1, 2)

    resistLithoImage.ImageType = '2d'
    resistLithoImage.SimulationType = 'resist'
    resistLithoImage.Intensity = intensityOutPut
    resistLithoImage.ImageX = torch.linspace(-mask.Period_X / 2, mask.Period_X / 2, waferNf)
    resistLithoImage.ImageY = torch.linspace(-mask.Period_Y / 2, mask.Period_Y / 2, waferNg)
    resistLithoImage.ImageZ = ImageZ_Resist
    
    return resistLithoImage


# Define a function to check the correctness of Calculate2DResistImage
def check():
    sr = Source()
    mk = Mask()  
    po = ProjectionObjective()  
    filmStack = FilmStack()
    rp = Receipe()  
    numerics = Numerics()  
    lithoImage = ImageData()

    # Call the function to be tested
    result = Calculate2DResistImage(sr, mk, po, filmStack, lithoImage, rp, numerics)
    
    # Print some validation information (you can add more checks)
    print("Intensity shape:", result.Intensity.shape)
    print("ImageX:", result.ImageX)
    print("ImageY:", result.ImageY)
    print("ImageZ:", result.ImageZ)

if __name__ == '__main__':
    # Call the check function to test Calculate1DAerialImage
    check()