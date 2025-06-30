import torch
import torch.sparse as sp
import sys, os

dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/../..")

from CalculateCharacteristicMatrix import CalculateCharacteristicMatrix
from litho.Numerics import Numerics
from litho.Source import Source
from litho.Receipe import Receipe
from litho.Mask import Mask
from litho.ProjectionObjective import ProjectionObjective


def cartesian_to_polar(x, y):
    
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    return rho, theta

def Calculate2DTCCMatrix(source, mask, projector, recipe, numerics):
    # Initialize data
    NA = projector.NA   # This line can be delete after merging
    if projector.LensType == 'Immersion':
        indexImage = projector.Index_ImmersionLiquid
    elif projector.LensType == 'Dry':
        indexImage = 1  # 1.44 
        if NA >= 1:    # Assuming NA is defined somewhere
            raise ValueError('Wrong NA!')
    else:
        raise ValueError('Unsupported Lens Type')

    wavelength = source.Wavelength
    xPitch = mask.Period_X
    yPitch = mask.Period_Y

    # Calculate shifted pupil function
    if numerics.ImageCalculationMode == 'scalar':
        SP, F_Valid, G_Valid, sourceData = CalculateShiftedPupilS(wavelength, projector, source, xPitch, yPitch, indexImage, recipe.Focus)
        TCCMatrix_Stacked = GetTCCMatrix(sourceData, SP)
    elif numerics.ImageCalculationMode == 'vector':
        SPXX, SPXY, SPYX, SPYY, SPXZ, SPYZ, F_Valid, G_Valid, sourceData = CalculateShiftedPupilV(wavelength, projector, source, xPitch, yPitch, indexImage, recipe.Focus)

        TCCMatrixX = GetTCCMatrix(sourceData, SPXX + SPYX)
        TCCMatrixY = GetTCCMatrix(sourceData, SPXY + SPYY)
        TCCMatrixZ = GetTCCMatrix(sourceData, SPXZ + SPYZ)
        TCCMatrix_Stacked = TCCMatrixX + TCCMatrixY + TCCMatrixZ
    TCCMatrix_Stacked = projector.Index_ImmersionLiquid * TCCMatrix_Stacked
    FG_ValidSize = [len(G_Valid), len(F_Valid)]
    
    return TCCMatrix_Stacked, FG_ValidSize

def CalculateShiftedPupilS(wavelength, projector, source, xPitch, yPitch, indexImage, focus):
    sourceData = source.Calc_SourceSimple()
    M = projector.Reduction
    NA = projector.NA
    normalized_xPitch = xPitch / (wavelength / NA)
    normalized_yPitch = yPitch / (wavelength / NA)
    Nf = torch.ceil(2 * normalized_xPitch).int()
    Ng = torch.ceil(2 * normalized_yPitch).int()
    f = (1 / normalized_xPitch) * torch.arange(-Nf, Nf + 1)
    g = (1 / normalized_yPitch) * torch.arange(-Ng, Ng + 1)
    ff, gg = torch.meshgrid(f, g)
    new_f = ff.flatten() + sourceData.X.flatten()
    new_g = gg.flatten() + sourceData.Y.flatten()
    theta, rho = torch.cartesian_to_polar(new_f, new_g)

    validPupil = (rho <= 1)
    validRho = rho[validPupil]
    validTheta = theta[validPupil]
    validRhoSquare = validRho.pow(2)

    obliquityFactor = torch.sqrt(torch.sqrt((1 - (M ** 2 * projector.NA ** 2) * validRhoSquare) / (1 - ((projector.NA / indexImage) ** 2) * validRhoSquare)))
    Orientation = 0
    aberration = projector.CalculateAberrationFast(validRho, validTheta, Orientation)  # You need to implement this function

    shiftedPupil = torch.zeros_like(validPupil)
    TempFocus = 1j * 2 * torch.pi / wavelength * torch.sqrt(indexImage ** 2 - NA ** 2 * validRhoSquare)
    shiftedPupil[validPupil] = obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * torch.exp(TempFocus * focus)
    shiftedPupil = torch.abs(shiftedPupil)
    return shiftedPupil, f, g, sourceData

def CalculateShiftedPupilV(wavelength, projector, source, xPitch, yPitch, indexImage, focus):
    sourceData = source.Calc_SourceSimple()  # You need to implement this function
    M = projector.Reduction
    NA = projector.NA
    normalized_xPitch = torch.tensor(xPitch / (wavelength / NA))
    normalized_yPitch = torch.tensor(yPitch / (wavelength / NA))
    Nf = torch.ceil(2 * normalized_xPitch).int()
    Ng = torch.ceil(2 * normalized_yPitch).int()
    f = (1 / normalized_xPitch) * torch.arange(-Nf, Nf + 1)
    g = (1 / normalized_yPitch) * torch.arange(-Ng, Ng + 1)
    ff, gg = torch.meshgrid(f, g)

    new_f = ff.reshape(-1, 1) + sourceData.X.reshape(1, -1)
    new_g = gg.reshape(-1, 1) + sourceData.Y.reshape(1, -1)
    rho, theta = cartesian_to_polar(new_f, new_g)
    rhoSquare = rho.pow(2)
    validPupil = (rho <= 1)
    validRho = rho[validPupil]
    validTheta = theta[validPupil]
    validRhoSquare = rhoSquare[validPupil]
    obliquityFactor = torch.sqrt(torch.sqrt((1 - (M ** 2 * projector.NA ** 2) * validRhoSquare) / (1 - ((projector.NA / indexImage) ** 2) * validRhoSquare)))
    Orientation = 0
    aberration = projector.CalculateAberrationFast(validRho, validTheta, Orientation)
    shiftedPupil = torch.zeros(validPupil.size()).to(torch.complex64)
    TempFocus = 1j * 2 * torch.pi / wavelength * torch.sqrt(indexImage ** 2 - NA ** 2 * validRhoSquare)
    shiftedPupil[validPupil] = obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * torch.exp(TempFocus * focus)
    
    M0xx, M0yx, M0xy, M0yy, M0xz, M0yz = CalculateCharacteristicMatrix(new_f, new_g, rhoSquare, NA, indexImage)
    rho_s, theta_s = cartesian_to_polar(sourceData.X, sourceData.Y)
    PolarizedX, PolarizedY = source.Calc_PolarizationMap(theta_s, rho_s)
    shiftedPupil = abs(shiftedPupil)

    SPXX = PolarizedX * M0xx * shiftedPupil
    SPXY = PolarizedX * M0xy * shiftedPupil
    SPXZ = PolarizedX * M0xz * shiftedPupil
    
    SPYX = PolarizedY * M0yx * shiftedPupil
    SPYY = PolarizedY * M0yy * shiftedPupil
    SPYZ = PolarizedY * M0yz * shiftedPupil

    return SPXX, SPXY, SPYX, SPYY, SPXZ, SPYZ, f, g, sourceData

def GetTCCMatrix(sourceData, shiftedPupil):
    n = sourceData.Value.size(0)  # Get the number of elements
    i = torch.arange(n)  # Create a tensor for row indices
    j = i.clone()  # Create a tensor for column indices (assuming it's a square matrix)
    # Convert i and j to tensors with dtype=torch.long
    
    # Create a sparse COO tensor
    S = torch.sparse_coo_tensor(
        torch.stack((i, j)),  # indices
        torch.complex(sourceData.Value, torch.zeros_like(sourceData.Value)),             # values
        size=(n, n),  # size of the sparse tensor
    )
    # Perform matrix operations
    TCCMatrix = torch.matmul(torch.matmul(shiftedPupil, S), shiftedPupil.t())  # Utilizing matrix conjugate transpose to get HSH*
    # Normalize the entire matrix by dividing it by the sum of all elements
    sum_value = torch.sum(sourceData.Value)
    TCCMatrix = TCCMatrix / sum_value

    return TCCMatrix

# Define a function to check the correctness of Calculate2DTCCMatrix
def check():
    sr = Source()
    mk = Mask()  # Initialize with appropriate values
    po = ProjectionObjective()  # Initialize with appropriate values
    rp = Receipe()  # Initialize with appropriate values
    numerics = Numerics()  # Initialize with appropriate values

    # Call the function to be tested
    tcc, fg = Calculate2DTCCMatrix(sr, mk, po, rp, numerics)
    # Print some validation information (you can add more checks)
    print(torch.sum(tcc))
    print("TCCMatrix:", tcc)
    print("FG:", fg)

if __name__ == '__main__':
    # Call the check function to test Calculate2DTCCMatrix
    check()