import torch
import sys, os

dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/../..")

from litho.Numerics import Numerics

def DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, numerics):
    maskNf = numerics.SampleNumber_Mask_X
    maskNg = numerics.SampleNumber_Mask_Y
    waferNf = numerics.SampleNumber_Wafer_X
    waferNg = numerics.SampleNumber_Wafer_Y
    
    U, S, _ = torch.svd(TCCMatrix_Stacked)
    if numerics.Hopkins_SettingType.lower() == 'order':
        socsNumber = min(numerics.Hopkins_Order, U.size(1) - 1)
    elif numerics.Hopkins_SettingType.lower() == 'threshold':
        rateThreshold = numerics.Hopkins_Threshold
        singularVector = torch.diag(S)
        totalSingular = torch.sum(singularVector)
        temp2 = 0
        socsNumber = U.size(1) - 1
        for i in range(len(singularVector) - 1):
            temp2 += singularVector[i]
            if temp2 > rateThreshold * totalSingular:
                socsNumber = i
                break
    else:
        raise ValueError('Error: TCCKernalSetting.method should be setNumber or setThreshold!')

    TCCMatrix_Kernel = torch.zeros(waferNg - 1, waferNf - 1, socsNumber)

    if waferNf > maskNf:
        rangeWaferNf = torch.arange((waferNf - maskNf + 2) // 2 - 1, (waferNf + maskNf - 2) // 2)
        rangeMaskNf = torch.arange(0, maskNf - 1)
    else:
        rangeWaferNf = torch.arange(0, waferNf - 1)
        rangeMaskNf = torch.arange((maskNf - waferNf + 2) // 2 - 1, (waferNf + maskNf - 2) // 2)
    
    if waferNg > maskNg:
        rangeWaferNg = torch.arange((waferNg - maskNg + 2) // 2 - 1, (waferNg + maskNg - 2) // 2)
        rangeMaskNg = torch.arange(0, maskNg - 1)
    else:
        rangeWaferNg = torch.arange(0, waferNg - 1)
        rangeMaskNg = torch.arange((maskNg - waferNg + 2) // 2 - 1, (waferNg + maskNg - 2) // 2)

    temp2 = torch.zeros(maskNg - 1, maskNf - 1)
    temp3 = torch.zeros(waferNg - 1, waferNf - 1)
    
    for i in range(socsNumber):
        temp1 = U[:, i].reshape(FG_ValidSize[1], FG_ValidSize[0])
        temp2[(maskNg - FG_ValidSize[1]) // 2 : (maskNg + FG_ValidSize[1]) // 2,
              (maskNf - FG_ValidSize[0]) // 2 : (maskNf + FG_ValidSize[0]) // 2] = temp1
        TCCMatrix_Kernel[:, :, i] = torch.fft.fftshift(temp2)

    diagS = S[0:socsNumber]
    diagS = diagS.reshape(1, 1, -1)
    TCCMatrix_Kernel = TCCMatrix_Kernel * torch.sqrt(diagS)

    return TCCMatrix_Kernel

# Define a function to check the correctness of DecomposeTCC_SOCS
def check():
    
    TCCMatrix_Stacked = torch.ones(529, 529)  # Example shape
    FG_ValidSize = (23, 23)
    
    numerics = Numerics()
    
    # Call the function to be tested
    result = DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, numerics)
    
    # Print the shape of the result for validation (you can add more checks)
    print("Result:", result)

if __name__ == '__main__':
# Call the check function to test DecomposeTCC_SOCS
    check()