import torch
import sys, os
import math

dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/../..")
# This class can be substituted by Numerics.py

from litho.Numerics import Numerics
from litho.FilmStack import FilmStack
from litho.Layer import Layer
from litho.Resist import Resist
from litho.Material import Material


def CalculateTransferMatrix(layers, sin_theta_inc, index_inc, wavelength, pol_mode):
    # Calculate transfer matrix
    
    # No antireflection coatings and antireflection films, TE transmission matrix
    M11 = torch.tensor(1.0)
    M12 = torch.tensor(0.0)
    M21 = torch.tensor(0.0)
    M22 = torch.tensor(1.0)
    
    if layers:  # If there are layers
        eta0 = math.sqrt(Numerics.Mu0 / Numerics.Epsilon0)
        for ly in layers:
            index_arc = ly.IndexComplex  # Complex value
            thickness_arc = ly.Thickness
            
            cos_theta_inner = torch.sqrt(1 - (sin_theta_inc * index_inc / index_arc)**2)
            eta_inner = eta0 / index_arc  # Non-magnetic medium approximation
            
            if pol_mode == 'TE':
                fai_inner = cos_theta_inner / eta_inner
            elif pol_mode == 'TM':
                fai_inner = -cos_theta_inner * eta_inner
            else:
                raise ValueError('Wrong polarization mode')
            
            ndk0ct = 2 * torch.pi * index_arc * thickness_arc * cos_theta_inner / wavelength
            jsin_ndk0ct = 1j * torch.sin(ndk0ct)  # Cannot be replaced by sqrt(1 - torch.sin(ndk0ct)**2)
            
            m11 = torch.cos(ndk0ct)
            m22 = m11
            m12 = jsin_ndk0ct / fai_inner
            m21 = jsin_ndk0ct * fai_inner

            ml11 = M11
            ml12 = M12
            ml21 = M21
            ml22 = M22
            # Multiply transfer matrices for multiple layers
            M11 = ml11 * m11 + ml12 * m21
            M12 = ml11 * m12 + ml12 * m22
            M21 = ml21 * m11 + ml22 * m21
            M22 = ml21 * m12 + ml22 * m22
    
    return M11, M12, M21, M22

if __name__ == '__main__':
    f = FilmStack()
    l1 = Layer(type='arc',
               thickness=400,
               material=Resist.AR165J())
    l2 = Layer(type='arc',
               thickness=200,
               material=Material.Silicon())
    l3 = Layer(type='arc',
               thickness=300,
               material=Resist.AR165J())
    l4 = Layer(type='arc',
               thickness=100,
               material=Material.Silicon())
    f = f.AddLayer(l2, 1)
    f = f.AddLayer(l1, 1)
    f = f.AddLayer(l4, 0)
    f = f.AddLayer(l3, 0)
    layers = f.GetTARCLayers()
    a = torch.pi / 4 / torch.ones(5, 5)
    sin_theta = torch.sqrt(1-a**2)
    print(sin_theta)
    index = 1.44
    wavelen = 193
    print(CalculateTransferMatrix(layers, sin_theta, index, wavelen, "TM"))


