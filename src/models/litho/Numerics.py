import math
import torch

class Numerics:
    SampleNumber_Source: int = 101
    SampleNumber_Mask_X: int = 81
    SampleNumber_Mask_Y: int = 81
    SampleNumber_Mask_RCWA_X: int = 11  # Sample number for RCWA Mask
    SampleNumber_Mask_RCWA_Y: int = 11

    # sampling number for mask calculatuon => mask / auto
    Sample_Calc_Mode: str = 'mask'

    SampleNumber_Wafer_X: int = 81
    SampleNumber_Wafer_Y: int = 81
    SampleNumber_Wafer_Z: int = 41
    # simulation settings
    SimulationRange_Aerial=0
    SimulationRange_Resist=[]
    # Normailization
    Normalization_Intensity=1
    # Imaging model => 'vector'  'sclar'
    ImageCalculationMode: str = 'vector'
    # Calculation model =>‘abbe’ ‘hopkins’
    ImageCalculationMethod: str = 'abbe'
    # Truncation method for hopkins model => 'order' 'Threshold'
    Hopkins_SettingType: str = 'order'
    Hopkins_Order=50
    Hopkins_Threshold=0.95  # (0, 1)
    # Resist model => physical / lumped / measured
    ResisitModel: str = 'physical'
    # Development model => Mack / Enhanced Mack / Notch
    DevelopmentModel: str = 'Mack'
    # Physical constant
    Mu0 = torch.tensor(4*math.pi*1e-7)  # Permeability => T·m/A
    Epsilon0 = torch.tensor(8.854187817*1e-12)  # Permittivity => F/m
    R = 0.0019876  # Gas constant => kcal/(K*mol)
    K0 = 273.15  # Absolute zero => K
    def __init__(self):
        pass


if __name__ == "__main__":
    numerics = Numerics()
    print(Numerics.SimulationRange_Aerial)
