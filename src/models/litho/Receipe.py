import torch

class Receipe:
    # Exposure settings
    Dose = 30  # Unit: mJ/cm2

    # focus setting
    # use the top surface of FilmStack as reference
    # negative -> wafer
    # positive -> projection objective
    Focus = torch.tensor([0])  # Unit: nm
    FocusRange = torch.tensor([0])
    FocusReference = 0  # Reference offset

    # PEB parameters
    PEB_Temperature = 120  # centigrade
    PEB_Time = 90  # PEB time -> second

    # Developing parameters
    def __init__(self):
        pass
        
