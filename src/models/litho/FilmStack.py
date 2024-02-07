import os,sys
dir_path = os.path.dirname(__file__)
sys.path.append(dir_path+"/..")

from litho.Layer import Layer
from litho.Material import Material
from litho.Resist import Resist

class FilmStack:
    def __init__(self):
        l1 = Layer(type='resist',
                   thickness=400,
                   material=Resist.AR165J())
        l2 = Layer(type='substrate',
                   thickness=float('inf'),
                   material=Material.Silicon())
        self.Layers = [l1, l2]

    def AddLayer(self, layer, pos):
        # insert layer at pos position
        # index increases from up to down
        layerLength = len(self.Layers)
        if ((pos > layerLength) | (pos < 0)):
            raise ValueError('Wrong layer position !')

        tempLayers = self.Layers
        self.Layers = []
        for iLayer in range(layerLength+1):
            if (iLayer < pos):
                self.Layers.append(tempLayers[iLayer])
            elif (iLayer == pos):
                self.Layers.append(layer)
            else:
                self.Layers.append(tempLayers[iLayer-1])
        return self

    def RemoveLayer(self, layerNo):
        layerLength = len(self.Layers)
        if ((layerNo >= layerLength) | (layerNo < 0)):
            raise ValueError('Wrong layer Number !')

        tempLayers = self.Layers
        self.Layers = []
        for iLayer in range(layerLength-1):
            if (iLayer < layerNo):
                iLayer2 = iLayer
            else:
                iLayer2 = iLayer + 1
            self.Layers.append(tempLayers[iLayer2])
        return self

    def GetNumberOfLayers(self):
        number = len(self.Layers)
        return number

    def GetTARCLayers(self):
        for i in range(len(self.Layers)):
            if (self.Layers[i].Type == 'resist'):
                break
        layers = []
        if (i == 0):
            pass
        elif (i < len(self.Layers) - 1):
            for j in range(i):
                layers.append(self.Layers[j])
                layers[j].IndexComplex = self.Layers[j].getRefractiveIndex()
        else:
            raise ValueError('No Resist!')
        return layers

    def GetResistLayer(self):
        for i in range(len(self.Layers)):
            if (self.Layers[i].Type == 'resist'):
                layer = self.Layers[i]
                return layer
        raise ValueError('No Resists Layer')

    def GetBARCLayers(self):
        iResist = []
        iSubstrate = []
        for i in range(len(self.Layers)):
            if (self.Layers[i].Type == 'resist'):
                iResist = i
            elif (self.Layers[i].Type == 'substrate'):
                iSubstrate = i

        layers = []
        if (iResist == []):
            raise ValueError('No Resist!')
        elif (iSubstrate == []):
            raise ValueError('No Substrate!')
        elif iSubstrate < iResist+1.5:
            pass
        elif iResist < iSubstrate:
            for i in range(iSubstrate-iResist-1):
                layers.append(self.Layers[iResist+1+i])
                layers[i].IndexComplex = \
                    self.Layers[iResist+1+i].getRefractiveIndex()
        else:
            raise ValueError('Wrong filmstack setting')
        return layers

    def GetFullARCLayers(self):
        layers = []
        for i in range(len(self.Layers) - 1):
            layers.append(self.Layers[i])
            layers[i].IndexComplex = self.Layers[i].getRefractiveIndex()
        return layers

    def GetResistIndex(self):
        for i in range(len(self.Layers)):
            if (i == len(self.Layers)-1):
                raise ValueError('Wrong resist setting')
            elif (self.Layers[i].Type == 'resist'):
                resistLayer = self.Layers[i]
                break
        index = resistLayer.getRefractiveIndex()
        return index

    def GetSubstrateIndex(self):
        substrateLayer = self.Layers[-1]
        if (substrateLayer.Type != 'substrate'):
            raise ValueError('Wrong substrate setting')
        index = substrateLayer.getRefractiveIndex()
        return index

    def GetResistThickness(self):
        for i in range(len(self.Layers)):
            if (i == len(self.Layers)-1):
                raise ValueError('Wrong resist setting')
            elif (self.Layers[i].Type == 'resist'):
                resistLayer = self.Layers[i]
                break
        thickness = resistLayer.Thickness
        return thickness


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
    n1 = f.GetNumberOfLayers()
    print(n1)
    f = f.RemoveLayer(4)
    n2 = f.GetNumberOfLayers()
    print(n2)
    for i in range(len(f.Layers)):
        print(f.Layers[i].__dict__)
    pass
