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

def Calculate2DAerialImage(source, mask, projector, lithoImage, receipe, numerics):
    mask_nf = numerics.SampleNumber_Mask_X
    mask_ng = numerics.SampleNumber_Mask_Y
    wafer_nf = numerics.SampleNumber_Wafer_X
    wafer_ng = numerics.SampleNumber_Wafer_Y

    source.PntNum = numerics.SampleNumber_Source
    sourceData = source.Calc_SourceSimple()
    weight = torch.sum(sourceData.Value)
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

    mask.Nf = mask_nf
    mask.Ng = mask_ng
    spectrum, mask_fs, mask_gs, _ = mask.CalculateMaskSpectrum(projector, source)

    SimulationRange = receipe.FocusRange - receipe.Focus

    Intensity = torch.zeros(wafer_nf, wafer_ng, len(SimulationRange))
    indexK = indexImage
    Orientation = mask.Orientation

    for iFocus in range(len(SimulationRange)):
        mask_fm, mask_gm = torch.meshgrid(mask_fs[:-1], mask_gs[:-1])
        intensity2D = torch.zeros(wafer_nf - 1, wafer_ng - 1, len(sourceData.Value))
        sourceX = sourceData.X
        sourceY = sourceData.Y
        sourceV = sourceData.Value

        focus = SimulationRange[iFocus]
        dfmdg = (mask_fs[1] - mask_fs[0]) * (mask_gs[1] - mask_gs[0])

        source_rho, source_theta = cartesian_to_polar(sourceX, sourceY)
        PolarizedX, PolarizedY = source.Calc_PolarizationMap(source_theta, source_rho)
        new_spectrum = spectrum[:-1, :-1]
        mask_fg2m = mask_fm ** 2 + mask_gm ** 2
        sourceXY2 = sourceX ** 2 + sourceY ** 2

        for j in range(len(sourceData.Value)):
            obliqueRaysMatrix = 1
            ExyzCalculateNumber_2D = 1

            if numerics.ImageCalculationMode.lower() == 'vector':
                ExyzCalculateNumber_2D = 3
            elif numerics.ImageCalculationMode.lower() == 'scalar':
                ExyzCalculateNumber_2D = 1

            rho2 = (mask_fg2m + 2 * (sourceX[j] * mask_fm + sourceY[j] * mask_gm) + sourceXY2[j]).to(torch.complex64)
            validPupil = torch.where(torch.real(rho2) <= 1)

            f_calc = mask_fm[validPupil] + sourceX[j]
            g_calc = mask_gm[validPupil] + sourceY[j]
            rho_calc, theta_calc = cartesian_to_polar(f_calc, g_calc)
            fgSquare = rho_calc ** 2

            obliquityFactor = torch.sqrt(torch.sqrt(
                (1 - (M ** 2 * NA ** 2) * fgSquare) / (1 - ((NA / indexImage) ** 2) * fgSquare)))
            
            aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, Orientation)
            pupilFilter = projector.CalculatePupilFilter(rho_calc, theta_calc)
            tempFocus = torch.exp(-1j * 2 * torch.pi / wavelength * torch.sqrt(indexK ** 2 - NA * NA * fgSquare) * focus)
            SpectrumCalc = new_spectrum[validPupil]

            TempHAber = SpectrumCalc * obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * pupilFilter * tempFocus
            
            
            if numerics.ImageCalculationMode == 'vector':
                obliqueRaysMatrix = torch.zeros(len(fgSquare), ExyzCalculateNumber_2D)
                m0xx, m0yx, m0xy, m0yy, m0xz, m0yz = CalculateCharacteristicMatrix(f_calc, g_calc, fgSquare, NA, indexImage)
                
                obliqueRaysMatrix[:, 0] = PolarizedX[j] * m0xx + PolarizedY[j] * m0yx
                obliqueRaysMatrix[:, 1] = PolarizedX[j] * m0xy + PolarizedY[j] * m0yy
                obliqueRaysMatrix[:, 2] = PolarizedX[j] * m0xz + PolarizedY[j] * m0yz
                
            rho2[:] = 0
            intensityTemp = torch.zeros(wafer_nf - 1, wafer_ng - 1)
            
            for iEM in range(ExyzCalculateNumber_2D):
                rho2[validPupil] = TempHAber * obliqueRaysMatrix[:, iEM]

                if wafer_nf == mask_nf and wafer_ng == mask_ng:
                    ExyzFrequency = rho2
                else:
                    ExyzFrequency = torch.zeros(wafer_nf - 1, wafer_ng - 1)
                    if wafer_nf > mask_nf:
                        rangeWaferNf = torch.arange((wafer_nf - mask_nf + 2) // 2, (wafer_nf + mask_nf - 2) // 2)
                        rangeMaskNf = torch.arange(0, mask_nf - 1)
                    else:
                        rangeWaferNf = torch.arange(0, wafer_nf - 1)
                        rangeMaskNf = torch.arange((mask_nf - wafer_nf + 2) // 2, (wafer_nf + mask_nf - 2) // 2)

                    if wafer_ng > mask_ng:
                        rangeWaferNg = torch.arange((wafer_ng - mask_ng + 2) // 2, (wafer_ng + mask_ng - 2) // 2)
                        rangeMaskNg = torch.arange(0, mask_ng - 1)
                    else:
                        rangeWaferNg = torch.arange(0, wafer_ng - 1)
                        rangeMaskNg = torch.arange((mask_ng - wafer_ng + 2) // 2, (wafer_ng + mask_ng - 2) // 2)

                    ExyzFrequency[rangeWaferNf, rangeWaferNg] = rho2[rangeMaskNf, rangeMaskNg]

                Exyz_Partial = torch.fft.fft2(ExyzFrequency)
                intensityTemp = intensityTemp + (torch.real(Exyz_Partial) ** 2 + torch.imag(Exyz_Partial) ** 2)

            intensity2D[:, :, j] = intensityTemp
        
        intensity2D = torch.reshape(sourceV, (1, 1, -1)) * intensity2D
        intensity2D = dfmdg ** 2 * torch.fft.fftshift(torch.sum(intensity2D, dim=2))
        intensity2D = torch.cat((intensity2D, intensity2D[:, 0].unsqueeze(1)), 1)
        intensity2D = torch.cat((intensity2D, intensity2D[0, :].unsqueeze(0)), 0)
        # intensity2D[:, wafer_nf - 1] = intensity2D[:, 0]
        # intensity2D[wafer_ng - 1, :] = intensity2D[0, :]
        intensity2D = torch.real(torch.rot90(intensity2D, 2))
        Intensity[:, :, iFocus] = (indexK / weight * intensity2D)
        Intensity = torch.transpose(Intensity,0,2)
    ImageX = torch.linspace(-mask.Period_X/2, mask.Period_X/2, wafer_nf)
    ImageY = torch.linspace(-mask.Period_Y/2, mask.Period_Y/2, wafer_ng)
    ImageZ = receipe.FocusRange
    
    lithoImage.ImageType = '2d'
    lithoImage.SimulatinType = 'aerial'
    lithoImage.Intensity = Intensity
    lithoImage.ImageX = ImageX
    lithoImage.ImageY = ImageY
    lithoImage.ImageZ = ImageZ

    return lithoImage

# Define a function to check the correctness of Calculate1DAerialImage
def check():
    sr = Source()
    mk = Mask.CreateMask('line_space')  # Initialize with appropriate values
    po = ProjectionObjective()  # Initialize with appropriate values
    rp = Receipe()  # Initialize with appropriate values
    numerics = Numerics()  # Initialize with appropriate values
    aerail_litho_image = ImageData()

    # Call the function to be tested
    result = Calculate2DAerialImage(sr, mk, po, aerail_litho_image, rp, numerics)
    
    # Print some validation information (you can add more checks)
    print("Intensity shape:", torch.sum(result.Intensity[:,:,0],0))
    print(result.Intensity.size())
    print("ImageX:", result.ImageX)
    print("ImageY:", result.ImageY)
    print("ImageZ:", result.ImageZ)

if __name__ == '__main__':
    # Call the check function to test Calculate1DAerialImage
    check()