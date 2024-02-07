import math

import torch
import torch.nn.functional as F
import sys
from torch.special import erf


def conv2(matr, ker, mode='same'):
    matr_x, matr_y = matr.size()
    ker_x, ker_y = ker.size()
    matr_4d = matr.reshape((1, 1, matr_x, matr_y))
    ker_4d = ker.reshape((1, 1, ker_x, ker_y))
    conv = F.conv2d(matr_4d, ker_4d, padding=mode)
    conv = conv.reshape((matr_x, matr_y))
    return conv


class Source:
    """Source.data is used for Abbe fomulation Source.mdata is used for
    Hopkins fomulation, Mutual Intensity, TCC calculation."""

    def __init__(self):
        self.PntNum = 101
        self.Wavelength = 193.368
        self.Shape = "annular"
        # Source shape :
        # 'annular''multipole''dipolecirc''quadrupole''MMA''pixel''quasar'

        self.SigmaOut = 0.8
        self.SigmaIn = 0.6

        self.SigmaCenter = 0.5
        self.SigmaRadius = 0.1
        self.OpenAngle = math.pi/6
        self.RotateAngle = 0
        self.PoleNumber = 2
        self.AdvancedParametersEnable = 0
        self.AdvancedParameters = AdvancedParameters()
        self.Source_Mask = None  # tensor

        # parameter for pixel source, maybe 2d tensor
        self.SPointX = torch.zeros(self.PntNum)
        self.SPointY = torch.zeros(self.PntNum)
        self.SPointValue = torch.zeros(self.PntNum)
        # PSF parameter
        # source blur parameter initialization
        self.PSFEnable = False
        self.PSFSigma = 0.02
        # source polarization
        self.PolarizationType = 't_pol'
        # Polorization mode :
        # 'x_pol'   'y_pol'  'unpolarized'   'r_pol'  't_pol' 'fun'
        # TODO 'line'
        self.PolarizationParameters = PolarizationParameters()
        # MMA model
        self.MMA = MMA()

        self.source_data = SourceData(self.PntNum, self.PntNum)  # 3d tensor

    def Calc_SourceSimple(self):
        self.source_data = self.Calc_SourceAll()
        Low_Weight = self.source_data.Value < 1e-5

        self.source_data.X = self.source_data.X[~Low_Weight]
        self.source_data.Y = self.source_data.Y[~Low_Weight]
        self.source_data.Value = self.source_data.Value[~Low_Weight]
        return self.source_data

    def Calc_SourceValid(self):
        data = self.Calc_SourceAll()
        self.source_data.Value = torch.where(
                (data.X.pow(2) + data.Y.pow(2)) > 1,
                0, data.Value
            )
        return self.source_data

    def Calc_SourceAll(self):
        Source_Coordinate_X = torch.linspace(-1, 1, self.PntNum)
        if ((self.Shape).lower() == "pixel"):
            # pixel source
            # Checking the mesh size of the source for pixel sources
            # to ensure that it conforms to the defined specifications.
            # FIXME: The coordinates and values of the pupil 
            #        need to be defined and the data format checked
            if (self.SPointX.numel() != self.PntNum**2):
                raise ValueError(
                    'Source Matrix Size X and difined PntNum are not matched '
                    )

            if (self.SPointY.numel() != self.PntNum**2):
                raise ValueError(
                    'Source Matrix Size Y and difined PntNum are not matched '
                    )

            if (self.SPointValue.numel() != self.PntNum**2):
                raise ValueError(
                    'Source Matrix Size Value and difined \
                    PntNum are not matched '
                    )
            self.source_data.X = self.SPointX
            self.source_data.Y = self.SPointY
            self.source_data.Value = self.SPointValue

        elif ((self.Shape).lower() == "annular"):
            if (self.AdvancedParametersEnable):  # whether to enable advanced parameter settings
                self.source_data = CalculateAnnularSourceMatrixWithAP(
                    self.SigmaOut, self.SigmaIn, Source_Coordinate_X,
                    Source_Coordinate_X, self.AdvancedParameters
                    )
            else:
                self.source_data = CalculateAnnularSourceMatrix(
                    self.SigmaOut, self.SigmaIn,
                    Source_Coordinate_X, Source_Coordinate_X
                    )

        elif ((self.Shape).lower() == "mma"):  # Micro-Mirror-Array model
            if (self.MMA.Coordinate.lower() == 'polar'):
                self.source_data = CalculateSourceMatrixWithMMA_V3(
                    self.MMA, Source_Coordinate_X, Source_Coordinate_X
                    )
            elif (self.MMA.Coordinate.lower() == 'cartesian'):
                self.source_data = CalculateSourceMatrixWithMMA_Cartesian(
                    self.MMA, Source_Coordinate_X, Source_Coordinate_X
                    )
            else:
                raise ValueError('Unsupported coordinate system')
            if self.MMA.Normalization:
                self.source_data.Value = self.source_data.Value / \
                    torch.max(self.source_data.Value)

        elif ((self.Shape).lower() == "quasar"):
            openAngle = self.OpenAngle
            rotateAngle = self.RotateAngle
            if (rotateAngle > math.pi/2) | (rotateAngle < -1 * math.pi / 2):
                raise ValueError(
                    'error: roate angle must be in the range of [-pi/2,pi/2] '
                    )
            if (openAngle <= 0) | (openAngle >= math.pi/2):
                raise ValueError(
                    'error: open angle must be in the range of [0,pi/2] '
                    )
            if self.AdvancedParametersEnable:  # whether to enable advanced parameter settings
                self.source_data = CalculateQuasarSourceMatrixWithAP(
                    self.SigmaOut, self.SigmaIn, openAngle, rotateAngle,
                    Source_Coordinate_X, Source_Coordinate_X,
                    self.AdvancedParameters
                    )
            else:
                self.source_data = CalculateQuasarSourceMatrix(
                    self.SigmaOut, self.SigmaIn, openAngle,
                    Source_Coordinate_X, Source_Coordinate_X
                    )

        elif ((self.Shape).lower() == "dipolecirc"):
            openAngle = self.OpenAngle
            rotateAngle = self.RotateAngle
            if (rotateAngle > math.pi/2) | (rotateAngle < -1 * math.pi / 2):
                raise ValueError(
                    'error: roate angle must be in the range of [-pi/2,pi/2] '
                    )
            if (openAngle < 0) | (openAngle > math.pi/2):
                raise ValueError(
                    'error: open angle must be in the range of [0,pi/2] '
                    )
            if self.AdvancedParametersEnable:  # whether to enable advanced parameter settings
                self.source_data = CalculateDipoleSourceMatrixWithAP(
                    self.SigmaOut, self.SigmaIn, openAngle, rotateAngle,
                    Source_Coordinate_X, Source_Coordinate_X,
                    self.AdvancedParameters
                    )
            else:
                self.source_data = CalculateDipoleSourceMatrix(
                    self.SigmaOut, self.SigmaIn, openAngle, rotateAngle,
                    Source_Coordinate_X, Source_Coordinate_X
                    )

        elif ((self.Shape).lower() == "multipole"):
            rotateAngle = self.RotateAngle
            if (rotateAngle > math.pi/2) | (rotateAngle < -1 * math.pi / 2):
                raise ValueError(
                    'error: roate angle must be in the range of [-pi/2,pi/2] '
                    )
            self.source_data = CalculateMultiCircSourceMatrix(
                self.SigmaCenter, self.SigmaRadius, self.PoleNumber,
                rotateAngle, Source_Coordinate_X, Source_Coordinate_X
                )

            if self.AdvancedParametersEnable:  # whether to enable advanced parameter settings
                raise Warning(
                    'Advanced Parameters are not \
                    supported for multipole source !'
                    )

        else:
            raise ValueError("unsupported illumination")

        sizeX, sizeY = self.source_data.Value.size()

        if (self.Source_Mask is not None):
            self.source_data.Value = torch.mul(
                self.source_data.Value, self.Source_Mask
                )
        # Data serialization and add source blur
        if (sizeX == sizeY):
            # Add source blur
            if self.PSFEnable:
                kernelSize = round(self.PntNum/10)*2 + 1
                kernelEdge = 1 / (self.PntNum - 1) * (kernelSize - 1)
                kernelX, kernelY = torch.meshgrid(
                    torch.linspace(
                        -kernelEdge, kernelEdge, kernelSize
                        ),
                    torch.linspace(
                        -kernelEdge, kernelEdge, kernelSize
                        )
                    )
                kernel = 1 / math.sqrt(2 * math.pi) / self.PSFSigma  \
                    * torch.exp(
                        - (kernelX.pow(2)+kernelY.pow(2)) /
                        self.PSFSigma ** 2
                        )
                kernel = kernel[~torch.all(kernel < 1e-6, 0)]
                kernel_tr = torch.transpose(kernel, 0, 1)
                kernel = kernel_tr[~torch.all(kernel_tr < 1e-6, 1)]
                kernel = kernel/torch.sum(kernel)
                self.source_data.Value = conv2(
                    self.source_data.Value, kernel, mode='same'
                    )
            # Set center point to 0
            self.source_data.Value[int((sizeX-1)/2), int((sizeY-1)/2)] = 0
            self.source_data = ConvertSourceMatrix2SourceData(self.source_data)
        return self.source_data

    # source polarization
    def Calc_PolarizationMap(self, theta, rho):
        if (self.PolarizationType == 'x_pol'):
            PolarizedX = torch.ones(theta.size())
            PolarizedY = torch.zeros(theta.size())
        elif (self.PolarizationType == 'y_pol'):
            PolarizedX = torch.zeros(theta.size())
            PolarizedY = torch.ones(theta.size())
        elif (self.PolarizationType == 'r_pol'):
            PolarizedX = torch.cos(theta)
            PolarizedY = torch.sin(theta)
        elif (self.PolarizationType == 't_pol'):
            PolarizedX = torch.sin(theta)
            PolarizedY = -1 * torch.cos(theta)
        elif (self.PolarizationType == 'line_pol'):
            PolarizedX = torch.mul(
                torch.sin(self.PolarizationParameters.Angle),
                torch.ones(theta.size())
                )
            PolarizedY = torch.mul(
                torch.cos(self.PolarizationParameters.Angle),
                torch.ones(theta.size())
                )
        elif (self.PolarizationType == 'fun'):
            PolarizedX = self.PolarizationParameters.PolFun_X(theta, rho)
            PolarizedY = self.PolarizationParameters.PolFun_Y(theta, rho)
        else:
            raise ValueError("unsupported polarization type")

        biz = rho < sys.float_info.epsilon
        if (len(biz) > sys.float_info.epsilon):
            PolarizedX[biz] = 0
            PolarizedY[biz] = 0
        return PolarizedX, PolarizedY


# MMA source
def CalculateSourceMatrixWithMMA_V3(
    MMA, Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    core_X = torch.mul(MMA.Rho, torch.cos(MMA.Theta))
    core_Y = torch.mul(MMA.Rho, torch.sin(MMA.Theta))
    pntNum = len(Source_Coordinate_X)
    core_Yr = (core_X * (pntNum + 1) / 2 + pntNum / 2).round()
    core_Xr = (core_Y * (pntNum + 1) / 2 + pntNum / 2).round()
    if (MMA.PSFType.lower() == 'gaussian'):
        PSFRange = math.ceil(MMA.PSFSigma * len(Source_Coordinate_X) + 1)
        PSFSigma2 = MMA.PSFSigma**2
        for iMMA in range(MMA.Number):
            Yrow = torch.arange(
                start=max(int(core_Xr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Xr[iMMA]) + PSFRange, pntNum),
                step=1)
            Xcol = torch.arange(
                start=max(int(core_Yr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Yr[iMMA]) + PSFRange, pntNum),
                step=1)
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.exp(-((s.X[Xcol[ix], Yrow] - core_X[iMMA]).pow(2) +
                                (s.Y[Xcol[ix], Yrow] - core_Y[iMMA]).pow(2))
                              / PSFSigma2)
    elif (MMA.PSFType.lower() == 'gaussianrect'):
        PSFRange = math.ceil(MMA.PSFSigma * len(Source_Coordinate_X) + 1)
        PSFSigma2 = MMA.PSFSigma**2
        for iMMA in range(MMA.Number):
            Yrow = torch.arange(
                start=max(int(core_Xr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Xr[iMMA]) + PSFRange, pntNum),
                step=1)
            Xcol = torch.arange(
                start=max(int(core_Yr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Yr[iMMA]) + PSFRange, pntNum),
                step=1)
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.mul(
                        torch.exp(-((s.X[Xcol[ix], Yrow] -
                                     core_X[iMMA]).pow(2)) / PSFSigma2),
                        torch.exp(-((s.Y[Xcol[ix], Yrow] -
                                     core_Y[iMMA]).pow(2)) / PSFSigma2))
    elif (MMA.PSFType.lower() == 'sinc2'):
        PSFRange = math.ceil(MMA.PSFSigma * len(Source_Coordinate_X) + 1)
        for iMMA in range(MMA.Number):
            Yrow = torch.arange(
                start=max(int(core_Xr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Xr[iMMA]) + PSFRange, pntNum),
                step=1)
            Xcol = torch.arange(
                start=max(int(core_Yr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Yr[iMMA]) + PSFRange, pntNum),
                step=1)
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.sinc(
                        torch.hypot(s.X[Xcol[ix], Yrow] - core_X[iMMA],
                                    s.Y[Xcol[ix], Yrow] - core_Y[iMMA])
                        / MMA.PSFSigma).pow(2)
    elif (MMA.PSFType.lower() == 'sinc2rect'):
        PSFRange = math.ceil(MMA.PSFSigma * len(Source_Coordinate_X) + 1)
        for iMMA in range(MMA.Number):
            Yrow = torch.arange(
                start=max(int(core_Xr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Xr[iMMA]) + PSFRange, pntNum),
                step=1)
            Xcol = torch.arange(
                start=max(int(core_Yr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Yr[iMMA]) + PSFRange, pntNum),
                step=1)
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.mul(
                        torch.sinc((s.X[Xcol[ix], Yrow] - core_X[iMMA])
                                   / MMA.PSFSigma),
                        torch.sinc((s.Y[Xcol[ix], Yrow] - core_Y[iMMA])
                                   / MMA.PSFSigma)
                        ).pow(2)
    elif (MMA.PSFType.lower() == 'sincrect'):
        PSFRange = math.ceil(MMA.PSFSigma * len(Source_Coordinate_X) + 1)
        for iMMA in range(MMA.Number):
            Yrow = torch.arange(
                start=max(int(core_Xr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Xr[iMMA]) + PSFRange, pntNum),
                step=1)
            Xcol = torch.arange(
                start=max(int(core_Yr[iMMA]) - PSFRange, 1) - 1,
                end=min(int(core_Yr[iMMA]) + PSFRange, pntNum),
                step=1)
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.abs(torch.mul(
                        torch.sinc((s.X[Xcol[ix], Yrow] - core_X[iMMA])
                                   / MMA.PSFSigma),
                        torch.sinc((s.Y[Xcol[ix], Yrow] - core_Y[iMMA])
                                   / MMA.PSFSigma)
                        ))
    elif (MMA.PSFType.lower() == 'bessel'):
        pass
    else:
        raise ValueError(
            'LithoModel:Source:ParameterError', 'Unsupported PSF Shape'
            )

    s.Value = MMA.PSFMAX * s.Value

    # Obtian the map of source sequence encoding by one-fourth positive encoding
    if ((MMA.SourceConstructType).lower() == 'full'):  # full source
        pass
    elif ((MMA.SourceConstructType).lower() == 'quarter'):  # Quarter decoding recovery
        s.Value = s.Value + torch.fliplr(s.Value)
        s.Value = s.Value + torch.flipud(s.Value)
    return s


def CalculateSourceMatrixWithMMA_Cartesian(
    MMA, Source_Coordinate_X, Source_Coordinate_Y,
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    pntNum = len(Source_Coordinate_X)
    core_Yr = (MMA.X * (pntNum + 1) / 2 + pntNum / 2).round()
    core_Xr = (MMA.Y * (pntNum + 1) / 2 + pntNum / 2).round()
    PSFRange = math.ceil(pntNum / 20)
    PSFSigma2 = MMA.PSFSigma**2
    for iMMA in range(MMA.Number):
        Yrow = torch.arange(
            start=max(int(core_Xr[iMMA]) - PSFRange, 1) - 1,
            end=min(int(core_Xr[iMMA]) + PSFRange, pntNum),
            step=1)
        Xcol = torch.arange(
            start=max(int(core_Yr[iMMA]) - PSFRange, 1) - 1,
            end=min(int(core_Yr[iMMA]) + PSFRange, pntNum),
            step=1)
        # Process the boundary source points separately
        if ((core_Xr[iMMA] == (pntNum + 1) / 2) |
           (core_Yr[iMMA] == (pntNum + 1) / 2)):
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.exp(-((s.X[Xcol[ix], Yrow] - MMA.X[iMMA]).pow(2) +
                                (s.Y[Xcol[ix], Yrow] - MMA.Y[iMMA]).pow(2))
                              / PSFSigma2
                              ) / 2
        else:
            for ix in range(len(Xcol)):
                s.Value[Xcol[ix], Yrow] = s.Value[Xcol[ix], Yrow] + \
                    torch.exp(-((s.X[Xcol[ix], Yrow] - MMA.X[iMMA]).pow(2) +
                                (s.Y[Xcol[ix], Yrow] - MMA.Y[iMMA]).pow(2))
                              / PSFSigma2)
    s.Value = MMA.PSFMAX * s.Value


    # Obtian the map of source sequence encoding by one-fourth positive encoding
    if ((MMA.SourceConstructType).lower() == 'full'):  # full source
        pass
    elif ((MMA.SourceConstructType).lower() == 'quarter'):  # Quarter decoding recovery
        s.Value = s.Value + torch.fliplr(s.Value)
        s.Value = s.Value + torch.flipud(s.Value)
    return s


# annular source
def CalculateAnnularSourceMatrixWithAP(
    SigmaOut, SigmaIn,
    Source_Coordinate_X, Source_Coordinate_Y,
    AdvancedParameters
):
    ap = AdvancedParameters
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    R2 = s.X.pow(2) + s.Y.pow(2)
    rho = torch.sqrt((s.X-ap.x0).pow(2) + (s.Y-ap.y0).pow(2))
    theta = torch.atan2(s.Y-ap.y0, s.X-ap.x0)
    ri = SigmaIn \
        + ap.dx0 / 2 * torch.cos(theta)\
        + ap.dy0 / 2 * torch.sin(theta)\
        + ap.cri2 * torch.cos(2 * theta)\
        + ap.sri2 * torch.sin(2 * theta)
    ro = SigmaOut \
        - ap.dx0 / 2 * torch.cos(theta)\
        - ap.dy0 / 2 * torch.sin(theta)\
        + ap.cro2 * torch.cos(2 * theta)\
        + ap.sro2 * torch.sin(2 * theta)
    Intensityazimuth = 1 - torch.mul(erf(ap.slo / SigmaOut * (rho-ro)),
                                     erf(ap.sli / SigmaIn * (rho-ri)))
    IntensityA = torch.ones(s.X.size())
    for iazm in range(1, 3):
        IntensityA = IntensityA \
            + torch.mul((ap.cA[iazm-1] * torch.cos(iazm * theta)
                        + ap.sA[iazm-1] * torch.sin(iazm * theta)),
                        (rho / SigmaOut).pow(iazm))
    s.Value = torch.mul(Intensityazimuth, IntensityA)
    s.Value = s.Value / torch.max(s.Value)
    s.Value[s.Value < ap.BG] = ap.BG
    s.Value[R2 > 1] = 0
    return s


def CalculateAnnularSourceMatrix(
    SigmaOut, SigmaIn,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    Radius = torch.sqrt(s.X.pow(2) + s.Y.pow(2))
    Radius_Out = ((Radius - SigmaOut) <= 1e-10).float()
    Radius_In = ((Radius - SigmaIn) >= -1e-10).float()
    s.Value = (Radius_Out + Radius_In) - 1
    return s


# quadrupole source
def CalculateQuasarSourceMatrixWithAP(
    SigmaOut, SigmaIn, openAngle, rotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y,
    AdvancedParameters
):
    ap = AdvancedParameters
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    R2 = s.X.pow(2) + s.Y.pow(2)
    rho = torch.sqrt((s.X-ap.x0).pow(2) + (s.Y-ap.y0).pow(2))
    theta = torch.atan2(s.Y-ap.y0, s.X-ap.x0)
    newtheta = theta - rotateAngle
    ri = SigmaIn \
        + ap.dx0 / 2 * torch.cos(newtheta)\
        + ap.dy0 / 2 * torch.sin(newtheta)\
        + ap.cri2 * torch.cos(2 * newtheta)\
        + ap.sri2 * torch.sin(2 * newtheta)
    ro = SigmaOut \
        - ap.dx0 / 2 * torch.cos(newtheta)\
        - ap.dy0 / 2 * torch.sin(newtheta)\
        + ap.cro2 * torch.cos(2 * newtheta)\
        + ap.sro2 * torch.sin(2 * newtheta)
    Intensityaxial = (1 - torch.mul(erf(ap.slo / SigmaOut * (rho - ro)),
                                    erf(ap.sli / SigmaIn * (rho - ri)))) / 2
    # Compute angular parameters
    newThetaFlag = (newtheta < -1 * math.pi)
    newtheta[newThetaFlag] = newtheta[newThetaFlag] + 2 * math.pi
    newThetaFlag = (newtheta > math.pi)
    newtheta[newThetaFlag] = newtheta[newThetaFlag] - 2 * math.pi
    newtheta = math.pi / 4 - torch.abs(
        torch.abs(torch.abs(newtheta) - math.pi/2) - math.pi/4
    )
    Intensityazimuth = (1 - torch.mul(
                            erf(ap.sla * (newtheta + openAngle / 2)),
                            erf(ap.sla * (newtheta - openAngle / 2)))
                        ) / 2
    IntensityA = torch.ones(s.X.size())
    for iazm in range(1, 3):
        IntensityA = IntensityA \
            + torch.mul((ap.cA[iazm-1] *
                         torch.cos(iazm * (theta - rotateAngle))
                        + ap.sA[iazm-1] *
                         torch.sin(iazm * (theta - rotateAngle))),
                        (rho/SigmaOut).pow(iazm))
    s.Value = torch.mul(torch.mul(Intensityaxial, IntensityA),
                        Intensityazimuth)
    if ap.A > sys.float_info.epsilon:
        s.Value = ap.A * s.Value/torch.max(s.Value)
    s.Value[s.Value < ap.BG] = ap.BG
    s.Value[R2 > 1] = 0
    return s


def CalculateQuasarSourceMatrix(
    SigmaOut, SigmaIn, openAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    Radius = torch.sqrt(s.X.pow(2) + s.Y.pow(2))
    theta = torch.atan2(s.Y, s.X)
    Indextheta1 = (torch.abs(theta) <= (openAngle / 2)) | \
        ((torch.abs(theta) >= (math.pi - openAngle / 2)) &
         (torch.abs(theta) <= math.pi))
    Indextheta2 = (torch.abs(theta - math.pi / 2) <= openAngle / 2) | \
        (torch.abs(theta + math.pi / 2) <= openAngle / 2)
    Indextheta = Indextheta1 | Indextheta2
    Index = (Radius <= SigmaOut) & (Radius >= SigmaIn) & Indextheta
    s.Value[Index] = 1
    return s


# dipolecirc source
def CalculateDipoleSourceMatrixWithAP(
    SigmaOut, SigmaIn, openAngle, rotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y,
    AdvancedParameters
):
    ap = AdvancedParameters
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    R2 = s.X.pow(2) + s.Y.pow(2)
    rho = torch.sqrt((s.X-ap.x0).pow(2) + (s.Y-ap.y0).pow(2))
    theta = torch.atan2(s.Y-ap.y0, s.X-ap.x0)
    newtheta = theta - rotateAngle
    ri = SigmaIn \
        + ap.dx0 / 2 * torch.cos(newtheta)\
        + ap.dy0 / 2 * torch.sin(newtheta)\
        + ap.cri2 * torch.cos(2 * newtheta)\
        + ap.sri2 * torch.sin(2 * newtheta)
    ro = SigmaOut \
        - ap.dx0 / 2 * torch.cos(newtheta)\
        - ap.dy0 / 2 * torch.sin(newtheta)\
        + ap.cro2 * torch.cos(2 * newtheta)\
        + ap.sro2 * torch.sin(2 * newtheta)
    Intensityaxial = (1 - torch.mul(erf(ap.slo / SigmaOut * (rho - ro)),
                                    erf(ap.sli / SigmaIn * (rho - ri)))) / 2
    newThetaFlag = (newtheta < -1 * math.pi)
    newtheta[newThetaFlag] = newtheta[newThetaFlag] + 2 * math.pi
    newThetaFlag = (newtheta > math.pi)
    newtheta[newThetaFlag] = newtheta[newThetaFlag] - 2 * math.pi
    newtheta = math.pi / 2 - torch.abs(torch.abs(newtheta) - math.pi / 2)
    Intensityazimuth = (1 - torch.mul(
                            erf(ap.sla * (newtheta + openAngle / 2)),
                            erf(ap.sla * (newtheta - openAngle / 2)))
                        ) / 2
    IntensityA = torch.ones(s.X.size())
    for iazm in range(1, 3):
        IntensityA = IntensityA \
            + torch.mul((ap.cA[iazm-1] *
                         torch.cos(iazm * (theta - rotateAngle))
                        + ap.sA[iazm-1] *
                         torch.sin(iazm * (theta - rotateAngle))),
                        (rho/SigmaOut).pow(iazm))
    s.Value = torch.mul(torch.mul(Intensityaxial, IntensityA),
                        Intensityazimuth)
    if (ap.A != 0):
        s.Value = ap.A * s.Value/torch.max(s.Value)
    s.Value[s.Value < ap.BG] = ap.BG
    s.Value[R2 > 1] = 0
    return s


def CalculateDipoleSourceMatrix(
    SigmaOut, SigmaIn, openAngle, rotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    Radius = torch.sqrt(s.X.pow(2) + s.Y.pow(2))
    theta = torch.atan2(s.Y, s.X)
    Indextheta = (torch.abs(torch.cos(theta - rotateAngle)) >=
                  math.cos(openAngle / 2))
    Index = (Radius <= SigmaOut) & (Radius >= SigmaIn) & Indextheta
    s.Value[Index] = 1
    return s


# multipule source
def CalculateMultiCircSourceMatrix(
    SigmaCenter, SigmaRadius, PoleNumber, RotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y)
    rotateStep = 2 * math.pi / PoleNumber
    for i in range(PoleNumber):
        xCenter = SigmaCenter * math.cos(RotateAngle + i * rotateStep)
        yCenter = SigmaCenter * math.sin(RotateAngle + i * rotateStep)
        Radius2 = (s.X - xCenter).pow(2) + (s.Y - yCenter).pow(2)
        Index = (Radius2 <= SigmaRadius**2)
        s.Value[Index] = 1
    return s

def ConvertSourceMatrix2SourceData(s):
    rSquare = s.X ** 2 + s.Y ** 2
    sizeSourceX = s.X.shape[0]
    # 将 rSquare 大于 1 的位置设为 0
    s.Value[rSquare > 1] = 0

    SourceData.X = torch.reshape(s.X, (sizeSourceX * sizeSourceX, 1))
    SourceData.Y = torch.reshape(s.Y, (sizeSourceX * sizeSourceX, 1))
    SourceData.Value = torch.reshape(s.Value, (sizeSourceX * sizeSourceX, 1))

    return SourceData

# data class
class SourceData:
    def __init__(self, x, y):
        self.X = torch.zeros((x, y))
        self.Y = torch.zeros((x, y))
        self.Value = torch.zeros((x, y))


# parameter class (advanced,polarization,mma)
class AdvancedParameters:
    def __init__(self):
        # Initialize advanced parameters 
        self.A = []  # Maximum relative intensity, [] doesn't participate in calculation
        self.BG = 0  # Background light intensity
        self.sli = 50  # Inner ring slope parameter
        self.slo = 50  # Outer ring slope parameter
        self.sla = 50  # Angular light intensity slope
        self.cA = [0, 0]
        self.sA = [0, 0]
        self.x0 = 0  # Ring center X offset
        self.y0 = 0  # Ring center Y offset
        self.dx0 = 0
        self.dy0 = 0
        self.cri2 = 0
        self.sri2 = 0
        self.cro2 = 0
        self.sro2 = 0


class PolarizationParameters:
    def __init__(self):
        self.Degree = 1  # Partially polarized light is not yet supported
        self.Angle = torch.tensor(0)  # Polarization direction: start at positive half of x-axis, counterclockwise
        self.PhaseDifference = 0
        # Used iff (PolarizationType == 'fun')
        self.PolFun_X = None  # xpol = fun(theta,rho)
        self.PolFun_Y = None  # ypol = fun(theta,rho)


class MMA:
    def __init__(self):
        self.Number = 128
        self.Coordinate = 'Polar'  # 'Cartesian'
        self.Rho = torch.sqrt(torch.rand(self.Number))
        self.Theta = math.pi/2 * torch.rand(self.Number)
        self.X = torch.rand(self.Number)*0.5
        self.Y = torch.rand(self.Number)*0.5
        self.PSFMAX = 0.01
        self.PSFSigma = 0.03
        self.PSFType = 'gaussian'  # gaussian / sinc / none
        self.Normalization = 0
        self.SourceConstructType = 'quarter'  # full / quarter


if __name__ == '__main__':

    # ############# test conv2() function  ##############
    # matr = (torch.linspace(1,49,49)).reshape((7,7))
    # ker = torch.ones((3,3))
    # conv = conv2(matr,ker,mode = "same")
    # print(conv)

    # ############Check class#########
    s = Source()

    # ##  Check Polarization() ##
    # # 'x_pol'   'y_pol'  'line_pol'   'r_pol'  't_pol' 'fun'
    # s.PolarizationType = 'r_pol'
    # rho = torch.tensor([0.15, 0.35, 0.55, 0.75, 0.95])
    # theta = math.pi/2 * \
    #     torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    # px, py = s.Calc_PolarizationMap(theta, rho)
    # print(px, py)

    # ##Check (s.Shape==pixel)  case##
    # ##and check Calc_SourceSimple & Calc_SourceValid ##
    # s.SPointX = 0.2 * torch.tensor(
    #     [[1, 1, 1, 1, 1, 1, 1, 1, 1],
    #      [2, 2, 2, 2, 2, 2, 2, 2, 2],
    #      [3, 3, 3, 3, 3, 3, 3, 3, 3],
    #      [4, 4, 4, 4, 4, 4, 4, 4, 4]]
    #     )
    # s.SPointY = 0.2 * torch.tensor(
    #     [[1, 2, 3, 4, 5, 6, 7, 8, 9],
    #      [1, 2, 3, 4, 5, 6, 7, 8, 9],
    #      [1, 2, 3, 4, 5, 6, 7, 8, 9],
    #      [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    #     )
    # s.SPointValue = 1e-6 * (torch.linspace(1,36,36)).resize_(4,9)
    # s.Shape = 'pixel'
    # s.PntNum = 6
    # print(s.SPointX.numel())
    # data = s.Calc_SourceValid()
    # print(data.X,data.Y,data.Value)

    # ##  Check (s.Shape==sometype)  case ##
    # s.Shape = 'mma'
    # s.PntNum = 9
    # s.MMA.Coordinate = 'Polar'
    # s.PSFEnable = 1
    # s.MMA.PSFType = 'sinc2'  # gaussian / sinc / none
    # s.MMA.Number = 5
    # s.MMA.Rho = torch.tensor([0.15, 0.35, 0.55, 0.75, 0.95])
    # s.MMA.Theta = math.pi/2 * \
    #     torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    # s.MMA.X = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45])
    # s.MMA.Y = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45])
    # s.RotateAngle = math.pi/4
    # data = s.Calc_SourceAll()
    # print(data.X,data.Y,data.Value)

    ##  Check (s.Shape==sometype)  case ##
    ##  sometype : 'annular' 'quasar' 'dipolecirc' 'multipole'

    data = s.Calc_SourceSimple()
    print(data.X.size(),data.Y.size(),data.Value.size())
    print(data.X,data.Y,data.Value)

    # ##  Check (s.Shape==sometype) with 'AP'  case ##
    # ##  sometype : 'annular' 'quasar' 'dipolecirc'
    # s.Shape = 'dipolecirc'
    # s.PntNum = 9
    # s.PoleNumber = 16
    # s.RotateAngle = math.pi/4
    # s.AdvancedParametersEnable = 1
    # s.AdvancedParameters.A = 1.5
    # s.AdvancedParameters.BG = 0.2
    # s.AdvancedParameters.sli = 50
    # s.AdvancedParameters.slo = 50
    # s.AdvancedParameters.sla = 50
    # s.AdvancedParameters.cA = [0.2, 0.2]
    # s.AdvancedParameters.sA = [0.2, 0.2]
    # s.AdvancedParameters.x0 = 0.2
    # s.AdvancedParameters.y0 = 0.2
    # s.AdvancedParameters.dx0 = 0.1
    # s.AdvancedParameters.dy0 = 0.1
    # s.AdvancedParameters.cri2 = 0.1
    # s.AdvancedParameters.sri2 = 0.1
    # s.AdvancedParameters.cro2 = 0.1
    # s.AdvancedParameters.sro2 = 0.1
    # data = s.Calc_SourceAll()
    # print(data.X,data.Y,data.Value)
