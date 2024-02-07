import math
import cmath
import torch
import torch.special
from scipy.spatial import Delaunay
import numpy as np
# from torchvision import transforms
from matplotlib import path
import torch.nn.functional as F
from PIL import Image, ImageDraw

# from gdsii.library import Library


from Source import Source
from ProjectionObjective import ProjectionObjective as PO

IMAGE_WH = 2048
class Mask:
    Spectrum = 1
    Fx = 0
    Gy = 0

    def __init__(self):
        self.MaskType = '2D'
        self.Period_X = 500
        self.Period_Y = 500
        self.Nf = 81
        self.Ng = 81
        self.Orientation = 0
        self.Background_Transmissivity = 0
        self.Background_Phase = 0
        self.MaskDefineType = '2D'
        self.Feature = []
        self.Cutline = []
        # openGDS
        self.gds_path: str = []
        self.layername: int = 11
        self.pixels_per_um: int = 100
        self.xmax=1024
        self.ymax=1024
        self.mask_groups = []
        self.x_gridsize=1
        self.y_gridsize=1
        self.CD=45

    def CalculateMaskSpectrum(self, po, sr):
        NA = po.NA
        Wavelength = sr.Wavelength
        HkBlank = None
        if (self.MaskType.lower() == '1d'):
            Hk, F, G =\
                self.__Mask1D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '1dpixel'):
            raise ValueError('To be implement')
        elif (self.MaskType.lower() == '2d'):
            Hk, F, G =\
                self.__Mask2D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '2dpixel'):
            Hk, F, G, HkBlank =\
                self.__PixelMask2D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '3d'):
            # TODO: The 3D mask algorithm is pending
            pass
        elif (self.MaskType.lower() == 'spectrum'):
            Hk = self.Spectrum
            F = self.Fx
            G = self.Gy
            HkBlank = self.Spectrum
        elif (self.MaskType.lower() == 'gds'):
            self.__openGDS()
            self.__maskfft()
            Hk = self.fdata

            normalized_Period_X = self.Period_X / (Wavelength/NA)
            normalized_Period_Y = self.Period_Y / (Wavelength/NA)
            F = (1/normalized_Period_X) * torch.arange(start=-(self.Nf-1)/2,
                                                    end=(self.Nf-1)/2 + 1,
                                                    step=1)
            G = (1/normalized_Period_Y) * torch.arange(start=-(self.Ng-1)/2,
                                                    end=(self.Ng-1)/2 + 1,
                                                    step=1)
        else:
            raise ValueError(
                'Error: There is not this type of mask!')
        return Hk, F, G, HkBlank

    def CalculateMaskSpectrumV2(self, po, sr):
        NA = po.NA
        Wavelength = sr.Wavelength
        if (self.MaskType.lower() == '1d'):
            Hk, F, G =\
                self.__Mask1D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '1dpixel'):
            raise ValueError('To be implement')
        elif (self.MaskType.lower() == '2d'):
            Hk, F, G =\
                self.__Mask2D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '2dpixel'):
            Hk, F, G, HkBlank =\
                self.__PixelMask2D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '3d'):
            # TODO: The 3D mask algorithm is pending
            pass
        elif (self.MaskType.lower() == 'spectrum'):
            Hk = self.Spectrum
            F = self.Fx
            G = self.Gy
        else:
            raise ValueError(
                'Error: There is not this type of mask!')
        return Hk, F, G

    def Convert2DiscreteMask(self):
        X = torch.linspace(-self.Period_X / 2, self.Period_X / 2, self.Nf)
        Y = torch.linspace(-self.Period_Y / 2, self.Period_Y / 2, self.Ng)
        X, Y = torch.meshgrid(X, Y)
        maskDiscreteData = torch.zeros(X.size())
        # Judge the position of all mask points
        inArea = torch.zeros(X.size())
        # What's the meaning of this sentence?
        inArea = (inArea == 1)
        # Mask background complex amplitude
        bgComplexAmplitude = self.Background_Transmissivity *\
            cmath.exp(1j * self.Background_Phase)
        featureNumber = len(self.Feature)
        for k in range(featureNumber):
            # The complex transmittance corresponding to the kth sprite
            am = self.Feature[k].ComplexAm[0] *\
                cmath.exp(1j * self.Feature[k].ComplexAm[1])
            # The sprites that make up the mask are rectangular
            if (self.Feature[k].ShapeType == 'r'):
                xVertexVector = [self.Feature[k].BoundaryVertexX[0],
                                 self.Feature[k].BoundaryVertexX[1],
                                 self.Feature[k].BoundaryVertexX[1],
                                 self.Feature[k].BoundaryVertexX[0]]
                yVertexVector = [self.Feature[k].BoundaryVertexY[0],
                                 self.Feature[k].BoundaryVertexY[0],
                                 self.Feature[k].BoundaryVertexY[1],
                                 self.Feature[k].BoundaryVertexY[1]]
                inRectangle = inpolygon_tensor(X, Y,
                                               xVertexVector,
                                               yVertexVector)
                # The overlapping part of the two indicates that it was calculated twice 
                # and needs to be removed once later
                # to avoid overlapping edges between sprites
                doubleCount = inArea & inRectangle
                # Remove the points in the area that have already been recorded,
                # and ensure that the inRectangle is the new point added each time
                inRectangle[doubleCount] = 0
                # Store the points where judgments have been made
                inArea = inArea | inRectangle
                # Accumulate to obtain the total mask pattern
                maskDiscreteData = maskDiscreteData +\
                    (am - bgComplexAmplitude) * inRectangle.float()

            # The sprites that make up the mask are triangle
            elif (self.Feature[k].ShapeType == 't'):
                xVertexVector = self.Feature[k].BoundaryVertexX
                yVertexVector = self.Feature[k].BoundaryVertexY
                inTriangle = inpolygon_tensor(X, Y,
                                              xVertexVector,
                                              yVertexVector)
                doubleCount = inArea & inTriangle
                inTriangle[doubleCount] = 0
                inArea = inArea | inTriangle
                maskDiscreteData = maskDiscreteData +\
                    (am - bgComplexAmplitude) * inTriangle.float()
            # The sprites that make up the mask are parallelogram
            elif (self.Feature[k].ShapeType == 'p'):
                xVertexVector = self.Feature[k].BoundaryVertexX
                yVertexVector = self.Feature[k]. BoundaryVertexY
                inParallelogram = inpolygon_tensor(X, Y,
                                                   xVertexVector,
                                                   yVertexVector)
                doubleCount = inArea & inParallelogram
                inParallelogram[doubleCount] = 0
                inArea = inArea | inParallelogram
                maskDiscreteData = maskDiscreteData +\
                    (am - bgComplexAmplitude) * inParallelogram.float()
            # The sprites that make up the mask are circles
            elif (self.Feature[k].ShapeType == 'c'):
                xCenterVector = self.Feature[k].BoundaryVertexX[0]
                yCenterVector = self.Feature[k].BoundaryVertexX[1]
                r = self.Feature[k]. BoundaryVertexY
                distance2 = (X - xCenterVector).pow(2) +\
                    (Y - yCenterVector).pow(2)
                distance = torch.sqrt(distance2)
                inCircle = ((distance - r) < 1e-6)
                doubleCount = inArea & inCircle
                inCircle[doubleCount] = 0
                inArea = inArea | inCircle
                maskDiscreteData = maskDiscreteData +\
                    (am - bgComplexAmplitude) * inCircle.float()

        # After integrating multiple sprites, they are added to the background
        # to obtain the complex transmittance of the complete mask
        maskDiscreteData = maskDiscreteData + bgComplexAmplitude

        self.Feature = maskDiscreteData
        self.MaskType = '2DPixel'
        return self

    def Convert2PSM(self, type, threshold):
        if (type == 'binary'):
            self.Feature = (torch.abs(self.Feature) > threshold[0]).int()
        elif (type == 'attpsm'):
            mkMap = (torch.abs(self.Feature) >
                     threshold[0]).int()
            imkMap = mkMap.to(torch.complex64)
            imkMap[mkMap < threshold[1]] = self.Background_Transmissivity *\
                cmath.exp(1j * self.Background_Phase)
            self.Feature = imkMap
        elif (type == 'altpsm'):
            mkMap = (torch.real(self.Feature) > threshold[0]).float()
            imkMap = mkMap.to(torch.complex64)
            imkMap[torch.real(self.Feature)
                   < threshold[1]] = cmath.exp(1j*math.pi)
            self.Feature = imkMap
        return self

    # private method
    def __Mask1D2Spectrum(self, NA, Wavelength):
        normalized_Frequency = NA / Wavelength  # Representation of coefficients before the Fourier transform
        bgComplexAmplitude = self.Background_Transmissivity *\
            cmath.exp(1j * self.Background_Phase)
        normalized_Period_X = self.Period_X / (Wavelength / NA)  # Normalize the sampling spacing
        f = (1/normalized_Period_X) * torch.arange(start=-(self.Nf - 1) / 2,
                                                   end=(self.Nf - 1) / 2 + 1,
                                                   step=1)
        g = 0
        Spectrum = torch.zeros(len(f))
        SubRectNum = len(self.Feature)
        for i in range(SubRectNum):
            if (self.Feature[i].ShapeType == 'r'):
                width = abs(self.Feature[i].BoundaryVertexX[1] -
                            self.Feature[i].BoundaryVertexX[0])  # width of the pattern
                am = self.Feature[i].ComplexAm[0] * \
                    cmath.exp(1j * self.Feature[i].ComplexAm[1])
                center = (self.Feature[i].BoundaryVertexX[1] +
                          self.Feature[i].BoundaryVertexX[0]) / 2  # The center position of the pattern
                temp = normalized_Frequency * (am - bgComplexAmplitude)\
                    * width * torch.mul(
                    torch.exp(-1j*2*math.pi*center*normalized_Frequency*f),
                    torch.sinc(width*normalized_Frequency*f))
                Spectrum = Spectrum + temp
        bgSpectrum = torch.zeros(len(f))  # Background spectrum
        bgSpectrum[torch.abs(f) < 1e-6] = \
            normalized_Period_X * bgComplexAmplitude
        Spectrum = Spectrum + bgSpectrum
        return Spectrum, f, g

    def __Mask2D2Spectrum(self, NA, Wavelength):
        bgComplexAmplitude = self.Background_Transmissivity *\
            cmath.exp(1j * self.Background_Phase)
        # Normalize the sampling spacing
        normalized_Period_X = self.Period_X / (Wavelength/NA)
        # The square refractive index has been taken into account in NA
        normalized_Period_Y = self.Period_Y / (Wavelength/NA)
        f = (1/normalized_Period_X) * torch.arange(start=-(self.Nf-1)/2,
                                                   end=(self.Nf-1)/2 + 1,
                                                   step=1)
        g = (1/normalized_Period_Y) * torch.arange(start=-(self.Ng-1)/2,
                                                   end=(self.Ng-1)/2 + 1,
                                                   step=1)
        normalized_Frequency = NA/Wavelength
        Hk = torch.zeros(len(g), len(f))
        SubRectNum = len(self.Feature)
        for ii in range(SubRectNum):
            if (self.Feature[ii].ShapeType == 'r'):  # rectangular pattern
                # Simplify design by only defining diagonal points
                xWidth = abs(self.Feature[ii].BoundaryVertexX[1]
                             - self.Feature[ii].BoundaryVertexX[0])
                yWidth = abs(self.Feature[ii].BoundaryVertexY[1]
                             - self.Feature[ii].BoundaryVertexY[0])
                a = self.Feature[ii].ComplexAm[0] *\
                    cmath.exp(1j * self.Feature[ii].ComplexAm[1])
                xCenter = (self.Feature[ii].BoundaryVertexX[0] +
                           self.Feature[ii].BoundaryVertexX[1]) / 2
                yCenter = (self.Feature[ii].BoundaryVertexY[0] +
                           self.Feature[ii].BoundaryVertexY[1]) / 2
                tempx = (torch.mul(
                    torch.exp(-1j*2*math.pi*xCenter*normalized_Frequency*f),
                    torch.sinc(xWidth*normalized_Frequency*f))).unsqueeze(0)
                tempy = (torch.mul(
                    torch.exp(-1j*2*math.pi*yCenter*normalized_Frequency*g),
                    torch.sinc(yWidth*normalized_Frequency*g))).unsqueeze(1)
                temp = (a-bgComplexAmplitude) * normalized_Frequency**2\
                    * xWidth * yWidth * (torch.mm(tempy, tempx))
                Hk = Hk + temp
            elif (self.Feature[ii].ShapeType == 'c'):  # Circular pattern
                # besselj
                a = self.Feature[ii].ComplexAm[0] *\
                    cmath.exp(1j * self.Feature[ii].ComplexAm[1])
                wr = self.Feature[ii].BoundaryVertexY[0]
                xCenter = self.Feature[ii].BoundaryVertexX[0]
                yCenter = self.Feature[ii].BoundaryVertexX[1]
                f2, g2 = torch.meshgrid(f, g)
                rho = torch.hypot(f2, g2)
                rho = rho + 1e-10
                tempx = (torch.exp(-1j * 2 * math.pi * xCenter
                                   * normalized_Frequency * f)).unsqueeze(0)
                tempy = (torch.exp(-1j * 2 * math.pi * yCenter
                                   * normalized_Frequency * g)).unsqueeze(1)
                temp2D = torch.mm(tempy, tempx)
                tempBessel = torch.div(
                    torch.special.bessel_j1(
                        2 * math.pi * wr * normalized_Frequency * rho),
                    rho)
                temp = (a - bgComplexAmplitude) *\
                    torch.mul(tempBessel, temp2D)
                Hk = Hk + temp
            elif (self.Feature[ii].ShapeType == 't'):  # Triangle pattern
                # Pay attention to the problem of normalization
                a = self.Feature[ii].ComplexAm[0] * \
                    cmath.exp(1j * self.Feature[ii].ComplexAm[1])
                trx1 = self.Feature[ii].BoundaryVertexX[0]
                trx2 = self.Feature[ii].BoundaryVertexX[1]
                trx3 = self.Feature[ii].BoundaryVertexX[2]

                try1 = self.Feature[ii].BoundaryVertexY[0]
                try2 = self.Feature[ii].BoundaryVertexY[1]
                try3 = self.Feature[ii].BoundaryVertexY[2]

                trxv = [trx1, trx2, trx3]
                tryv = [try1, try2, try3]
                temp = Mask.triMask2spectrum(trxv, tryv, a, bgComplexAmplitude,
                                             normalized_Frequency, f, g)
                Hk = Hk + temp
            elif (self.Feature[ii].ShapeType == 'p'):  # single point
                a = self.Feature[ii].ComplexAm[0] *\
                    cmath.exp(1j * self.Feature[ii].ComplexAm[1])
                polyX = self.Feature[ii].BoundaryVertexX
                polyY = self.Feature[ii].BoundaryVertexY
                dt = delaunayTriangulation(polyX, polyY)
                for iipoly in range(dt.size(0)):
                    trxv = [polyX[dt[iipoly, 0]],
                            polyX[dt[iipoly, 1]],
                            polyX[dt[iipoly, 2]]]
                    tryv = [polyY[dt[iipoly, 0]],
                            polyY[dt[iipoly, 1]],
                            polyY[dt[iipoly, 2]]]
                    graycenter = [sum(trxv)/len(trxv), sum(tryv)/len(tryv)]
                    inone = inpolygon(graycenter[0], graycenter[1],
                                      polyX, polyY)
                    if inone:
                        temp = Mask.triMask2spectrum(
                            trxv, tryv, a, bgComplexAmplitude,
                            normalized_Frequency, f, g
                            )
                        Hk = Hk + temp
        bool_tensor = torch.mm((torch.abs(g) < 1e-9).unsqueeze(1).int(),
                               (torch.abs(f) < 1e-9).unsqueeze(0).int())
        value = (normalized_Period_X*normalized_Period_Y) * bgComplexAmplitude
        BkSpectrum = torch.where(bool_tensor == 1, value, 0)
        Hk = Hk + BkSpectrum
        Hk = torch.t(Hk)
        return Hk, f, g

    def __PixelMask2D2Spectrum(self, NA, Wavelength):
        h = self.Feature
        h = h.to(torch.complex64)
        PntNum_Y, PntNum_X = h.size()  # PntNum is odd

        if (self.Nf != PntNum_X):
            raise ValueError(
                'error: mask data size 2 must be equal to Mask.Nf ')
        if (self.Ng != PntNum_Y):
            raise ValueError(
                'error: mask data size 1 must be equal to Mask.Ng ')

        period_X = self.Period_X
        period_Y = self.Period_Y
        Start_X = -self.Period_X / 2
        Start_Y = -self.Period_Y / 2

        normPitchX = period_X / (Wavelength / NA)
        normPitchY = period_Y / (Wavelength / NA)
        normSX = Start_X / (Wavelength / NA)
        normSY = Start_Y / (Wavelength / NA)

        f = (1/normPitchX) * torch.arange(start=-(PntNum_X - 1) / 2,
                                          end=(PntNum_X - 1) / 2 + 1,
                                          step=1)
        g = (1/normPitchY) * torch.arange(start=-(PntNum_Y - 1) / 2,
                                          end=(PntNum_Y - 1) / 2 + 1,
                                          step=1)

        normdx = normPitchX / (PntNum_X - 1)
        normdy = normPitchY / (PntNum_Y - 1)

        normDfx = 1 / normPitchX
        normDfy = 1 / normPitchY
        normfx = normDfx * torch.arange(start=-(PntNum_X - 1) / 2,
                                        end=(PntNum_X - 1) / 2 + 1,
                                        step=1)
        normfy = normDfy * torch.arange(start=-(PntNum_Y - 1) / 2,
                                        end=(PntNum_Y - 1) / 2 + 1,
                                        step=1)
        nDFTVectorX = torch.linspace(0, PntNum_X - 1, PntNum_X)
        nDFTVectorY = torch.linspace(0, PntNum_Y - 1, PntNum_Y)

        # AFactor: PntNum_X rows PntNum_Y columns
        AFactor1 = torch.mm(
            torch.exp(-1j * 2 * math.pi * normSY * normfy).unsqueeze(1),
            torch.exp(-1j * 2 * math.pi * normSX * normfx).unsqueeze(0)
        )
        AFactor = (normdx * normdy) * AFactor1

        hFactor = torch.mm(
            torch.exp(1j * math.pi * nDFTVectorY).unsqueeze(1),
            torch.exp(1j * math.pi * nDFTVectorX).unsqueeze(0)
        )
        hh = torch.mul(h, hFactor)

        h2D = hh[0:-1, 0:-1]
        HYEnd = hh[-1, 0:-1]
        HXEnd = hh[0:-1, -1]
        HXYEnd = hh[-1, -1]

        DFTH_2D = torch.fft.fft2(h2D)
        DFTH_2D = torch.cat((DFTH_2D, DFTH_2D[:, 0].unsqueeze(1)), 1)
        DFTH_2D = torch.cat((DFTH_2D, DFTH_2D[0].unsqueeze(0)), 0)

        DFTHYEnd = torch.fft.fft(HYEnd)
        DFTHYEnd = torch.cat((DFTHYEnd, DFTHYEnd[0].unsqueeze(0)), 0)
        DFTHM_12D = torch.mm(
            torch.ones((PntNum_Y, 1), dtype=torch.complex64),
            DFTHYEnd.unsqueeze(0)
        )

        DFTHXEnd = torch.fft.fft(HXEnd)
        DFTHXEnd = torch.cat((DFTHXEnd, DFTHXEnd[0].unsqueeze(0)), 0)
        DFTHN_12D = torch.mm(
            DFTHXEnd.unsqueeze(1),
            torch.ones((1, PntNum_X), dtype=torch.complex64)
        )

        Hk = torch.mul(
            AFactor,
            DFTH_2D + DFTHM_12D + DFTHN_12D + HXYEnd
        )

        normalized_Frequency = NA / Wavelength
        tempx = torch.sinc(self.Period_X * normalized_Frequency * f)
        tempy = torch.sinc(self.Period_Y * normalized_Frequency * g)
        HkBlank = normalized_Frequency**2 * self.Period_X * \
            self.Period_Y * torch.mm(tempy.unsqueeze(1), tempx.unsqueeze(0))
        return Hk, f, g, HkBlank

    # def __openGDS(self):
        # gdsdir = self.gds_path
        # layername = self.layername
        # pixels_per_um = self.pixels_per_um

        # with open(gdsdir, "rb") as stream:
        #     lib = Library.load(stream)

        # a = lib.pop(0)
        # b = []
        # xmin = []
        # xmax = []
        # ymin = []
        # ymax = []
        # for ii in range(0, len(a)):
        #     if a[ii].layer == layername:
        #         if len(a[ii].xy) > 1:
        #             aa = np.array(a[ii].xy) / 1000 * pixels_per_um
        #             b.append(aa)
        #             xmin.append(min([k for k, v in aa]))
        #             xmax.append(max([k for k, v in aa]))
        #             ymin.append(min([v for k, v in aa]))
        #             ymax.append(max([v for k, v in aa]))
        # self.polylist = b

        # xmin = min(xmin)
        # xmax = max(xmax)
        # ymin = min(ymin)
        # ymax = max(ymax)

        # center_x = (xmax - xmin) // 2
        # center_y = (ymax - ymin) // 2

        # cpoints = []

        # cx_r = np.arange(center_x, xmax, IMAGE_WH // 2)
        # cx_l = -np.arange(-center_x, -xmin, IMAGE_WH // 2)
        # cxs = np.hstack((cx_l, cx_r))

        # cy_u = np.arange(center_y, ymax, IMAGE_WH // 2)
        # cy_d = -np.arange(-center_y, -ymin, IMAGE_WH // 2)
        # cys = np.hstack((cy_d, cy_u))

        # for x in cxs:
        #     for y in cys:
        #         cpoints.append((x, y))

        # cpoints = list(set(cpoints))


        # for cc in cpoints:
        #     self.xmin = cc[0] - (IMAGE_WH // 2)
        #     self.xmax = self.xmin + IMAGE_WH
        #     self.ymin = cc[1] - (IMAGE_WH // 2)
        #     self.ymax = self.ymin + IMAGE_WH
        #     self.x_range = [self.xmin, self.xmax]
        #     self.y_range = [self.ymin, self.ymax]

        #     self.Nf = int((self.xmax - self.xmin) / self.x_gridsize)
        #     self.Ng = int((self.ymax - self.ymin) / self.y_gridsize)
        #     img = Image.new("L", (self.Nf, self.Ng), 0)

        #     self.perimeter = 0.0
        #     for ii in self.polylist:
        #         pp = np.array(ii)  # polygon
        #         polygonlen = len(pp)
        #         self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))

        #         pp[:, 0] = (pp[:, 0] - self.xmin) / self.x_gridsize
        #         pp[:, 1] = (pp[:, 1] - self.ymin) / self.y_gridsize
        #         vetex_list = list(pp)
        #         polygon = [tuple(y) for y in vetex_list]
        #         ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        #         self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))

        #     self.mask_groups.append(torch.from_numpy(np.array(img)))

        # self.data = self.mask_groups[0]
        # # Fourier transform pair, pyfftw syntax
        # self.spat_part = torch.zeros((self.Ng, self.Nf), dtype=torch.complex128)
        # self.freq_part = torch.zeros((self.Ng, self.Nf), dtype=torch.complex128)

    def __maskfft(self):
        self.spat_part[:] = torch.fft.ifftshift(self.data)
        self.freq_part = torch.fft.fftn(self.spat_part)
        self.fdata = torch.fft.fftshift(self.freq_part)

    def CreateSpaceMask(self, spaceCD, spacePitch):
        self.MaskType = '1D'
        self.bgTransmission = 0.0
        self.bgPhase = 0
        self.Background_Transmissivity = 0
        self.Background_Phase = 0
        self.Period_X = spacePitch  # 1000 at the beginning
        self.Period_Y = spacePitch  # 1000 at the beginning

        self.Feature = [Feature(shapetype='r',
                        boundaryvertexX=[-spaceCD/2, spaceCD/2],
                        boundaryvertexY=[-spacePitch/2, spacePitch/2],
                        complexam=[1, 0])]
        # Set cutline
        self.Cutline = [Cutline(x=[-spacePitch/2, spacePitch/2],
                                y=[0, 0])]
        return self

    def CreateLineMask(self, lineCD, linePitch):
        self.MaskType = '1D'
        self.bgTransmission = 0.0
        self.bgPhase = 0
        self.Background_Transmissivity = 1
        self.Background_Phase = 0
        self.Period_X = linePitch  # 1000 at the beginning
        self.Period_Y = linePitch  # 1000 at the beginning

        self.Feature = [Feature(shapetype='r',
                        boundaryvertexX=[-lineCD/2, lineCD/2],
                        boundaryvertexY=[-linePitch/2, linePitch/2],
                        complexam=[0, 0])]

        # Set cutline
        self.Cutline = [Cutline(x=[-linePitch/2, linePitch/2],
                                y=[0, 0])]
        return self

    @staticmethod
    def CreateMask(maskType, varargin=[]):
        mask = Mask()
        length = len(varargin)
        if (maskType.lower() == 'line'):
            if (length == 0):
                lineCD = 45
                linePitch = 90
            elif (length == 1):
                lineCD = varargin[0]
                linePitch = lineCD * 2
            elif (length == 2):
                lineCD = varargin[0]
                linePitch = varargin[1]
            mask = mask.CreateLineMask(lineCD, linePitch)
        elif (maskType.lower() == 'space'):
            if (length == 0):
                spaceCD = 45
                spacePitch = 90
            elif (length == 1):
                spaceCD = varargin[0]
                spacePitch = spaceCD * 2
            elif (length == 2):
                spaceCD = varargin[0]
                spacePitch = varargin[1]
            mask = mask.CreateSpaceMask(spaceCD, spacePitch)
        elif (maskType.lower() == 'space_alt'):
            if (length == 0):
                spaceCD = 45
                spacePitch = 90
            elif (length == 1):
                spaceCD = varargin[0]
                spacePitch = spaceCD * 2
            elif (length == 2):
                spaceCD = varargin[0]
                spacePitch = varargin[1]

            maskXL = spacePitch * 2
            maskYL = spacePitch * 2

            mask.MaskType = '1D'
            mask.Background_Transmissivity = 0
            mask.Background_Phase = 0
            mask.Period_X = maskXL  # 1000 at the beginning
            mask.Period_Y = maskYL  # 1000 at the beginning

            mask.Feature.append(
                Feature(shapetype='r',
                        boundaryvertexX=[-(spacePitch+spaceCD)/2,
                                         -(spacePitch-spaceCD)/2],
                        boundaryvertexY=[-maskYL/2, maskYL/2],
                        complexam=[1, 0])
                )
            mask.Feature.append(
                Feature(shapetype='r',
                        boundaryvertexX=[(spacePitch-spaceCD)/2,
                                         (spacePitch+spaceCD)/2],
                        boundaryvertexY=[-maskYL/2, maskYL/2],
                        complexam=[1, math.pi])
                )

            # Set cutline
            mask.Cutline = [Cutline(x=[0, maskXL],
                                    y=[0, 0])]
        else:
            mask.bgTransmission = 0.0
            mask.bgPhase = 0
            mask.Background_Transmissivity = 0
            mask.Background_Phase = 0
            if (maskType.lower() == 'crossgate'):
                mask.Period_X = 420  # 1000 at the beginning
                mask.Period_Y = 420  # 1000 at the beginning
                boundaryVertexX = [[-180, -135],
                                   [-75, -30],
                                   [-75, -30],
                                   [30, 75],
                                   [-30, 75],
                                   [30, 75],
                                   [135, 180]]
                boundaryVertexY = [[-210, 210],
                                   [-25, 210],
                                   [-210, -70],
                                   [-210, -25],
                                   [-25, 25],
                                   [70, 210],
                                   [-210, 210]]
                cutlineX = [[0, 105],
                            [0, 0],
                            [-105, 0],
                            [105, 210]]
                cutlineY = [[100, 100],
                            [-100, 100],
                            [100, 100],
                            [0, 0]]
                # rectangle
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
                # Set cutline
                for i in range(len(cutlineX)):
                    mask.Cutline.append(
                        Cutline(x=cutlineX[i],
                                y=cutlineY[i])
                        )
            elif (maskType.lower() == 'contact_holes'):
                mask.Period_X = 210  # 1000 at the beginning
                mask.Period_Y = 210  # 1000 at the beginning
                boundaryVertexX = [[30, 75],
                                   [-75, -30],
                                   [30, 75],
                                   [-75, -30]]
                boundaryVertexY = [[30, 75],
                                   [-75, -30],
                                   [-75, -30],
                                   [30, 75]]
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            elif (maskType.lower() == 'line_space'):
                mask.Period_X = 720  # 1000 at the beginning
                mask.Period_Y = 720  # 1000 at the beginning
                boundaryVertexX = [[-22.5, 22.5],
                                   [67.5, 112.5],
                                   [157.5, 202.5],
                                   [-112.5, -67.5],
                                   [-202.5, -157.5]]
                boundaryVertexY = [[-300, 300],
                                   [-300, 300],
                                   [-300, 300],
                                   [-300, 300],
                                   [-300, 300]]
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            elif (maskType.lower() == 'complex'):
                mask.bgTransmission = 0.0
                mask.bgPhase = 0
                mask.Period_X = 1200  # 1000 at the beginning
                mask.Period_Y = 1200  # 1000 at the beginning
                boundaryVertexX = [[-510, 510],
                                   [240, 465],
                                   [240, 345],
                                   [120, 345],
                                   [120, 345],
                                   [240, 345],
                                   [240, 465],
                                   [-345, -120],
                                   [-345, -240],
                                   [-465, -240],
                                   [-465, -240],
                                   [-345, -240],
                                   [-345, -120],
                                   [-510, 510],
                                   [-345, -120],
                                   [-345, -240],
                                   [-465, -240],
                                   [-465, -240],
                                   [-345, -240],
                                   [-345, -120],
                                   [240, 465],
                                   [240, 345],
                                   [120, 345],
                                   [120, 345],
                                   [240, 345],
                                   [240, 465],
                                   [-510, 510]]
                boundaryVertexY = [[427.5, 472.5],
                                   [337.5, 382.5],
                                   [292.5, 337.5],
                                   [247.5, 292.5],
                                   [157.5, 202.5],
                                   [112.5, 157.5],
                                   [67.5, 112.5],
                                   [247.5, 292.5],
                                   [292.5, 337.5],
                                   [337.5, 382.5],
                                   [67.5, 112.5],
                                   [112.5, 157.5],
                                   [157.5, 202.5],
                                   [-22.5, 22.5],
                                   [-202.5, -157.5],
                                   [-157.5, -112.5],
                                   [-112.5, -67.5],
                                   [-382.5, -337.5],
                                   [-337.5, -292.5],
                                   [-292.5, -247.5],
                                   [-112.5, -67.5],
                                   [-157.5, -112.5],
                                   [-202.5, -157.5],
                                   [-292.5, -247.5],
                                   [-337.5, -292.5],
                                   [-382.5, -337.5],
                                   [-472.5, -427.5]]
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            elif (maskType.lower() == 'mypattern'):
                mask.bgPhase = 0
                mask.Period_X = 600  # 1000 at the beginning
                mask.Period_Y = 600  # 1000 at the beginning
                boundaryVertexX = [[-200, 200],
                                   [172, 200],
                                   [-200, -140],
                                   [-80, -20],
                                   [-200, 100],
                                   [-200, -172],
                                   [-132, -104],
                                   [-64, -36],
                                   [72, 100]]
                boundaryVertexY = [[172, 200],
                                   [-200, 172],
                                   [90, 118],
                                   [90, 118],
                                   [-200, -172],
                                   [-172, 0],
                                   [-172, 0],
                                   [-172, 0],
                                   [-172, -50]]
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            elif (maskType.lower() == 'sram'):
                maskSize = 400
                mask.bgTransmission = 0.0
                mask.bgPhase = 0
                mask.Period_X = maskSize  # 1000 at the beginning
                mask.Period_Y = maskSize  # 1000 at the beginning
                mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=[-510, 510],
                                boundaryvertexY=[-472.5, -427.5],
                                complexam=[1, 0])
                        )
            else:
                raise ValueError(
                    'error:This type of mask has not been included')

        return mask

    @staticmethod
    def CreateparameterizedMask(maskType, varargin=[]):
        mask = Mask()
        length = len(varargin)
        if (maskType.lower() == 'line'):
            if (length == 0):
                lineCD = 45
                linePitch = 90
            elif (length == 1):
                lineCD = varargin[0]
                linePitch = lineCD * 2
            elif (length == 2):
                lineCD = varargin[0]
                linePitch = varargin[1]
            mask = mask.CreateLineMask(lineCD, linePitch)
        elif (maskType.lower() == 'space'):
            if (length == 0):
                spaceCD = 45
                spacePitch = 90
            elif (length == 1):
                spaceCD = varargin[0]
                spacePitch = spaceCD * 2
            elif (length == 2):
                spaceCD = varargin[0]
                spacePitch = varargin[1]
            mask = mask.CreateLineMask(spaceCD, spacePitch)
        elif (maskType.lower() == 'space_end_dense'):
            if (length == 0):
                gapCD = 45
                spaceCD = 45
                spacePitch = 1000
            elif (length == 1):
                gapCD = varargin[0]
                spaceCD = 45
                spacePitch = spaceCD * 2
            elif (length == 2):
                gapCD = varargin[0]
                spaceCD = varargin[1]
                spacePitch = spaceCD * 2
            elif (length == 3):
                gapCD = varargin[0]
                spaceCD = varargin[1]
                spacePitch = varargin[2]

            mask.bgTransmission = 0.0
            mask.bgPhase = 0
            mask.Background_Transmissivity = 0
            mask.Background_Phase = 0
            mask.Period_X = spacePitch  # default: 90
            mask.Period_Y = spacePitch  # default: 90

            mask.Feature.append(
                Feature(shapetype='r',
                        boundaryvertexX=[-mask.Period_X/2, -gapCD/2],
                        boundaryvertexY=[-spaceCD/2, spaceCD/2],
                        complexam=[1, 0])
            )
            mask.Feature.append(
                Feature(shapetype='r',
                        boundaryvertexX=[gapCD/2, mask.Period_X/2],
                        boundaryvertexY=[-spaceCD/2, spaceCD/2],
                        complexam=[1, 0])
            )
        elif (maskType.lower() == 'space_end_t'):
            pass
        return mask

    # TODO
    @staticmethod
    def triMask2spectrum(trxv, tryv, ComplexAm,
                         BgComplexAm, normFre, f, g):
        # ff, gg = torch.meshgrid(f, g)
        # newf2 = (trxv[1] - trxv[0]) * normFre * ff +\
        #     (tryv[1] - tryv[0]) * normFre * gg
        # newg2 = (trxv[2] - trxv[0]) * normFre * ff +\
        #     (tryv[2] - tryv[0]) * normFre * gg

        # detTri = normFre*normFre*(
        #     (trxv[1] - trxv[0]) * (tryv[2] - tryv[0])
        #     - (trxv[2] - trxv[0]) * (tryv[1] - tryv[0]))
        # triPhase = torch.exp(-1j * 2 * math.pi * (trxv[0] * normFre * ff +
        #                                           tryv[0] * normFre * gg))

        # Hfg = unit_tri_fft(newf2, newg2)
        # temp = detTri * triPhase * Hfg
        # temp = (ComplexAm-BgComplexAm) * temp
        temp = None
        return temp


class Feature:
    def __init__(self,
                 shapetype,
                 boundaryvertexX,
                 boundaryvertexY,
                 complexam):
        self.ShapeType = shapetype
        self.BoundaryVertexX = boundaryvertexX
        self.BoundaryVertexY = boundaryvertexY
        self.ComplexAm = complexam


class Cutline:
    def __init__(self, x, y):
        self.X = x
        self.Y = y


# create triangles by point set (polyX,polyY)
def delaunayTriangulation(polyX, polyY):
    points = torch.zeros(len(polyX), 2)
    for i in range(len(polyX)):
        points[i] = torch.tensor([polyX[i], polyY[i]])
    dt_array = Delaunay(points)
    dt = torch.tensor(dt_array.simplices)
    return dt


# Judge if (xq,yq) in polygon created by xv,yv
# xq, yq are float
def inpolygon(xq, yq, xv, yv):
    q = [(xq, yq)]
    p = path.Path([(xv[i], yv[i]) for i in range(len(xv))])
    inpoly = (p.contains_points(q, radius=1e-10))[0]
    return inpoly
# xq,yq are tensor
def inpolygon_tensor(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(len(xv))])
    inpoly_tensor = p.contains_points(q, radius=1e-10).reshape(shape)
    inpoly_tensor = torch.tensor(inpoly_tensor)
    return inpoly_tensor


if __name__ == '__main__':
    # # check delaunayTraiangulation function
    # polyX = [0, 1, 2, 3, 4, 5]
    # polyY = [1.1, 2.2, 4.4, 8.8, 16, 32]
    # dt = delaunayTriangulation(polyX, polyY)
    # print(dt)
    # print(dt.size(0))
    # print([polyY[dt[2, 0]],
    #        polyY[dt[2, 1]],
    #        polyY[dt[2, 2]]])

    # # check inpolygon_tensor function
    # a = torch.linspace(1, 25, 25).reshape(5, 5)
    # a[0, 0] = 0
    # b = torch.linspace(25, 1, 25).reshape(5, 5)
    # b[0, 0] = 0
    # xv = [0, 15, 15, 0]
    # yv = [0, 0, 15, 15]
    # inpo = inpolygon_tensor(a, b, xv, yv)
    # print(inpo)

    # # check 'CreateMask', 'CreateLineMask', 'CreateSpaceMask'
    # # 'CreateparameterizedMask' function
    # varargin = []
    # type = 'space_end_dense'
    # mask = Mask.CreateparameterizedMask(type, varargin)
    # print(mask.__dict__)
    # print('\n')
    # for i in range(len(mask.Feature)):
    #     print(mask.Feature[i].__dict__)
    #     print('\n')
    # for i in range(len(mask.Cutline)):
    #     print(mask.Cutline[i].__dict__)
    #     print('\n')

    # check 'CalculateMaskSpectrumV2', 'CalculateMaskSpectrum'
    # which contains 3 private functions
    po = PO()
    sr = Source()
    varargin=[]
    type = 'crossgate'
    mask = Mask.CreateMask(type,varargin)
    mask.MaskType = '2d'
    Hk, f, g, _ = mask.CalculateMaskSpectrum(po, sr)
    spectrum = Hk.real.detach().numpy()
    print(spectrum)
    np.savetxt('mask.csv', spectrum.real, delimiter=',')
    # # check 'Convert2DiscreteMask'
    # mask = Mask.CreateMask('crossgate')
    # mask = mask.Convert2DiscreteMask()
    # print(mask.__dict__)
    # print(mask.Feature[35])

    # # check 'Convert2PSM'
    # mask = Mask()
    # mask.Feature = torch.linspace(1, 25, 25).reshape(5, 5)
    # type = 'altpsm'
    # threshold = [3, 0.2]
    # mask.Background_Transmissivity = 0.2
    # mask.Background_Phase = 1
    # mask = mask.Convert2PSM(type, threshold)
    # print(mask.__dict__)
    pass
