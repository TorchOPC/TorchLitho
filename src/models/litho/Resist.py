# from lithoByWZ.Material import Material
import os, sys
dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/..")
from litho.Material import Material
import math



class Resist(Material):
    def __init__(self):
        Material.__init__(self)
        self.Type: str = 'novolac'  # 'novolac' / 'CAR'
        # PEB: post exposure bake
        self.PEB_A: int = 0     # unit:1/um  Chemical enlarging adhesive A=0
        self.PEB_B = None        # unit:1/um  JSR AR165J 1.1（Calculated by material k）
        self.PEB_C: float = 0.05  # unit:1/um cm2/mJ JSR AR165J 0.0434
        self.PEB_LnAr = None  # diffusion coefficient: nm2/s
        self.PEB_Ea = None  # activation energy: kcal/mol

        self.Development_Rmax = None  # maximum development rate
        self.Development_Rmin = None  # minimum development rate
        self.Development_Mth = None  # PAC concentration threshold
        self.Development_Rresin = None
        self.Development_n = None
        self.Development_l = None  # inhibition reaction order
        self.Development_Rrs = None  # surface reaction relative rate
        self.Development_Din = None  # depth of surface inhibition effect
        self.Tone: str = 'Positive'  # Photoresist properties: 'Positive' / 'Negative'

    def get_PEB_B(self):
        value = 4 * math.pi * self.k / (self.wavelength / 1e3)
        return value

    def set_PEB_B(self, value):
        self.k = value * (self.wavelength / 1e3) / (4 * math.pi)

    @staticmethod
    def AR165J():
        r = Resist()
        r.name = 'AR165J'
        r.n = 1.7135
        r.k = 0.016894
        r.wavelength = 193

        r.PEB_A = 0
        r.PEB_C = 0.0434

        r.PEB_LnAr = 27.12  # nm2/s
        r.PEB_Ea = 19.65  # activation energy: kcal/mol

        r.Development_Rmax = 100
        r.Development_Rmin = 0.1
        r.Development_Rresin = 3.5
        r.Development_Mth = -100
        r.Development_n = 3
        r.Development_Rrs = 0.6
        r.Development_Din = 120
        return r


if __name__ == "__main__":
    r = Resist()
    print(r.__dict__)
    r = Resist.AR165J()
    r.set_PEB_B(100)
    print(r.get_PEB_B())
