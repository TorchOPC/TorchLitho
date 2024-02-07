import os, sys
dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/..")

from litho.Material import Material
from litho.Resist import Resist


class Layer:
    def __init__(self, type='', thickness=0, material=Material()):
        self.Type = type  # "arc" or "resist" or "substrate" no case requriment
        self.Thickness = thickness  # unit: nm
        self.Material = material  # Type: Material or Resist
        self.IndexComplex = 0  # Complex index

    def getRefractiveIndex(self):
        index = self.Material.n + self.Material.k * 1j
        return index


if __name__ == '__main__':
    l1 = Layer(type='resist',
               thickness=400,
               material=Resist.AR165J())
    print(l1.Type)
    print(l1.Thickness)
    print(l1.Material)
    print(l1.getRefractiveIndex())
