import torch
import sys, os

dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/../..")

from litho.Numerics import Numerics
from litho.Source import Source
from litho.Receipe import Receipe
from litho.Mask import Mask
from litho.ProjectionObjective import ProjectionObjective

def CalculateAerialImage_SOCS(mask, TCCMatrix_SOCS, source, projector, numerics):
    # Get Image
    maskNf = numerics.SampleNumber_Mask_X  # default 81
    maskNg = numerics.SampleNumber_Mask_Y  # default 81

    waferNf = numerics.SampleNumber_Wafer_X  #  default 81
    waferNg = numerics.SampleNumber_Wafer_Y  #  default 81

    spectrum, f, g, _ = mask.CalculateMaskSpectrum(projector, source)  # Assuming CalculateMaskSpectrum is defined

    if waferNf == maskNf and waferNg == maskNg:
        spectrumEx = spectrum[:-1, :-1]
    else:
        if waferNf > maskNf:
            rangeWaferNf = range((waferNf - maskNf + 2) // 2, (waferNf + maskNf - 2) // 2)
            rangeMaskNf = range(0, maskNf - 1)
        else:
            rangeWaferNf = range(0, waferNf - 1)
            rangeMaskNf = range((maskNf - waferNf + 2) // 2, (waferNf + maskNf - 2) // 2)

        if waferNg > maskNg:
            rangeWaferNg = range((waferNg - maskNg + 2) // 2, (waferNg + maskNg - 2) // 2)
            rangeMaskNg = range(0, maskNg - 1)
        else:
            rangeWaferNg = range(0, waferNg - 1)
            rangeMaskNg = range((maskNg - waferNg + 2) // 2, (waferNg + maskNg - 2) // 2)

        spectrumEx = torch.zeros(waferNf - 1, waferNg - 1)
        spectrumEx[rangeWaferNf, rangeWaferNg] = spectrum[rangeMaskNf, rangeMaskNg]

    temp = TCCMatrix_SOCS * torch.fft.fftshift(spectrumEx).unsqueeze(2)
    temp = temp.permute(2, 1, 0)
    Etemp = (f[1] - f[0]) * (g[1] - g[0]) * torch.fft.fft2(temp)
    Esquare = torch.abs(Etemp)**2
    intensity = torch.sum(Esquare, dim=0)
    intensity = torch.fft.fftshift(intensity)
    intensity = torch.cat((intensity, intensity[:, 0].unsqueeze(1)), dim = 1)
    intensity = torch.cat((intensity, intensity[0, :].unsqueeze(0)), dim = 0)
    intensity = torch.rot90(intensity, 2)

    return intensity

# Define a function to check the correctness of CalculateSOCS
def check():
    sr = Source()
    mk = Mask()  # Initialize with appropriate values
    mk.CreateLineMask(45, 90)
    po = ProjectionObjective()  # Initialize with appropriate values
    numerics = Numerics()  # Initialize with appropriate values
    matrix = torch.ones(80,80,50)
    # Call the function to be tested
    result = CalculateAerialImage_SOCS(mk, matrix, sr, po, numerics)
    
    # Print some validation information (you can add more checks)
    print("Intensity sum:", torch.sum(result))
    print("Intensity:",result)
if __name__ == '__main__':
    # Call the check function to test CalculateSOCS
    check()