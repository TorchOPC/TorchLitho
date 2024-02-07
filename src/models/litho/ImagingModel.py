import os, sys
dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/..")
sys.path.append(dir_path+"/functions")
from litho.Numerics import Numerics
from litho.Source import Source
from litho.Receipe import Receipe
from litho.Mask import Mask
from litho.ProjectionObjective import ProjectionObjective
from litho.FilmStack import FilmStack
from litho.ImageData import ImageData
from litho.functions.Calculate1DAerialImage import Calculate1DAerialImage
from litho.functions.Calculate1DResistImage import Calculate1DResistImage
from litho.functions.Calculate2DAerialImage import Calculate2DAerialImage
from litho.functions.Calculate2DResistImage import Calculate2DResistImage
from litho.functions.Calculate2DTCCMatrix import Calculate2DTCCMatrix
from litho.functions.CalculateTransferMatrix import CalculateTransferMatrix
from litho.functions.DecomposeTCC_SOCS import DecomposeTCC_SOCS
from litho.functions.CalculateNormalImage import CalculateNormalImage
from litho.functions.CalculateAerialImage_SOCS import CalculateAerialImage_SOCS
import torch
import numpy as np


class ImagingModel:
    def __init__(self):
        self.Numerics = Numerics()
        self.Source = Source()
        self.Mask = Mask.CreateMask('line_space')
        self.Projector = ProjectionObjective()
        self.FilmStack = FilmStack()
        self.LithoImageProtype = ImageData()
        self.Receipe = Receipe()

    def CalculateAerialImage(self):
        sr = self.Source
        mk = self.Mask
        po = self.Projector
        rp = self.Receipe
        nm = self.Numerics
        lip = self.LithoImageProtype
        if (self.Mask.MaskType == '1D'):
            ali = Calculate1DAerialImage(sr, mk, po, lip, rp, nm)
        elif (self.Mask.MaskType == '2D') \
                | (self.Mask.MaskType == '2DPixel'):
            if (nm.ImageCalculationMethod == 'abbe'):
                ali = Calculate2DAerialImage(sr, mk, po, lip, rp, nm)
            elif (nm.ImageCalculationMethod == 'hopkins'):
                ali = lip
                TCCMatrix_Stacked, FG_ValidSize = \
                    Calculate2DTCCMatrix(sr, mk, po, rp, nm)
                TCCMatrix_Kernel = \
                    DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, nm)
                ali.Intensity = CalculateAerialImage_SOCS(mk, TCCMatrix_Kernel,
                                                          sr, po, nm)
                ali.ImageX = torch.linspace(-mk.Period_X / 2,
                                            mk.Period_X / 2,
                                            nm.SampleNumber_Wafer_X)
                ali.ImageY = torch.linspace(-mk.Period_Y / 2,
                                            mk.Period_Y / 2,
                                            nm.SampleNumber_Wafer_Y)
                ali.ImageZ = rp.FocusRange
                ali.ImageType = '2d'
                ali.SimulationType = 'aerial'
            else:
                raise ValueError('Unsupported Calculation Method')
        else:
            raise ValueError('Unsupported Mask')
        if nm.Normalization_Intensity:
            ni = CalculateNormalImage(sr, mk, po, rp, nm)
            ali.Intensity = ali.Intensity / ni
        return ali

    def CalculateResistImage(self):
        sr = self.Source
        mk = self.Mask
        po = self.Projector
        fs = self.FilmStack
        rp = self.Receipe
        nm = self.Numerics
        lip = self.LithoImageProtype
        if (self.Mask.MaskType == '1D'):
            rli = Calculate1DResistImage(sr, mk, po, fs, lip, rp, nm)
        elif (self.Mask.MaskType == '2D') \
                | (self.Mask.MaskType == '2DPixel'):
            if (nm.ImageCalculationMethod == 'abbe'):
                rli = Calculate2DResistImage(sr, mk, po, fs, lip, rp, nm)
            elif (nm.ImageCalculationMethod == 'hopkins'):
                rli = lip
                TCCMatrix_Stacked, FG_ValidSize =\
                    Calculate2DTCCMatrix(sr, mk, po, fs, rp, nm)
                TCCMatrix_Kernel = DecomposeTCC_SOCS(
                    TCCMatrix_Stacked, FG_ValidSize, nm)
                rli.Intensity = CalculateAerialImage_SOCS(mk, TCCMatrix_Kernel,
                                                          sr, po, nm)
                rli.ImageX = torch.linspace(-mk.Period_X / 2,
                                            mk.Period_X / 2,
                                            nm.SampleNumber_Wafer_X)
                rli.ImageY = torch.linspace(-mk.Period_Y / 2,
                                            mk.Period_Y / 2,
                                            nm.SampleNumber_Wafer_Y)
                rli.ImageZ = rp.FocusRange
                rli.ImageType = '2d'
                rli.SimulationType = 'aerial'
            else:
                raise ValueError('Unsupported Calculation Method')
        else:
            raise ValueError('Unsupported Mask')

        if nm.Normalization_Intensity:
            ni = CalculateNormalImage(sr, mk, po, rp, nm)
            rli.Intensity = rli.Intensity / ni
        return rli

    def CalculateExposedLatentImage(self, rli = None):
        if (rli == None):
            rli = self.CalculateResistImage()
        else:
            pass
        fs = self.FilmStack
        rp = self.Receipe
        nm = self.Numerics
        eli = self.CalculateExposedLatentImage(rli, fs, rp, nm)
        return eli




def check():
    im = ImagingModel()
    im.Numerics.ImageCalculationMethod = "abbe"
    # result = im.CalculateAerialImage()
    # result = im.CalculateResistImage()
    im.Numerics.ImageCalculationMethod = "hopkins"
    im.Mask = Mask.CreateMask('line_space')
    # im.Mask.CreateLineMask(45,90)
    # print(im.Mask.__dict__)
    result = im.CalculateAerialImage() 
    # result = im.CalculateResistImage()


    a = torch.real(result.Intensity).contiguous().view(-1, 81)
    a = a.detach().numpy()
    np.savetxt('ImagingModel.csv',a,delimiter=',')
    # sr = Source()
    # mk = Mask()  
    # po = ProjectionObjective()  
    # filmStack = FilmStack()
    # rp = Receipe()  
    # numerics = Numerics()  
    # lithoImage = ImageData()

    # # Call the function to be tested
    # result = Calculate2DResistImage(sr, mk, po, filmStack, lithoImage, rp, numerics)
    
    # Print some validation information (you can add more checks)

    print("Intensity shape:", result.Intensity)
    print("ImageX:", result.ImageX)
    print("ImageY:", result.ImageY)
    print("ImageZ:", result.ImageZ)

if __name__ == '__main__':
    # Call the check function to test Calculate1DAerialImage
    check()
