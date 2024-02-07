import torch


class ImagingView:
    @staticmethod
    def PlotMaskSpectrum(mask, source, projector):
        Spectrum, fx, fy, Hb = mask.CalculateMaskSpectrum(projector, source)
        if (mask.MaskType == '1D') | (mask.MaskType == '1Dpixel'):
            view_fx = fx
            valid_fx = (torch.abs(view_fx) < 2)
            view_fx = view_fx[valid_fx]
            view_fy = torch.zeros(view_fx.size())
            absSpectrum = torch.abs(Spectrum)
            view_Spectrum = absSpectrum[valid_fx]
            view_Spectrum = 2 * view_Spectrum / torch.max(view_Spectrum)
            # quiver3(view_fx,view_fy,zeros(size(view_fx)),zeros(size(view_fx)),zeros(size(view_fx)),view_Spectrum,0)
        elif (mask.MaskType == '2D') | (mask.MaskType == '2Dpixel'):
            view_fx, view_fy = torch.meshgrid(fx, fy)
            valid_fx = (torch.abs(view_fx) < 2)
            valid_fy = (torch.abs(view_fy) < 2)
            validfxy = valid_fx & valid_fy
            view_fx2 = view_fx[validfxy]
            view_fy2 = view_fy(validfxy)
            absSpectrum = torch.abs(Spectrum)
            view_Spectrum = absSpectrum[validfxy]
            view_Spectrum = 2 * view_Spectrum / torch.max(view_Spectrum)
            # quiver3(view_fx2,view_fy2,zeros(size(view_fx2)),zeros(size(view_fx2)),zeros(size(view_fx2)),view_Spectrum,0);
        else:
            pass
        # rectangle('Position',[-1,-1,2,2],'Curvature',[1,1],'EdgeColor','r','LineWidth',1.5)
        # axis equal
        # box on
        # xlabel('Frequency X')
        # ylabel('Frequency Y')
        # zlabel('Amplitude')
        # title('Mask Spectrum')
        # set(gca,'fontsize',13)
        # set(gca,'FontName','Times New Roman')

    @staticmethod
    def PlotPupilPattern(mask, source, projector):
        pass

    @staticmethod
    def PlotMask(mk, varargin):
        pass

    @staticmethod
    def PlotSource(source, view_size):
        pass

    @staticmethod
    def PlotSource_MMA_PSF(source, view_size):
        pass

    @staticmethod
    def PlotSource_MMA_Center(source):
        pass

    @staticmethod
    def PlotPolarization(source):
        pass

    @staticmethod
    def PlotAberration(projector):
        pass

    @staticmethod
    def PlotAerialImage(lithoImage):
        pass

    @staticmethod
    def PlotResistImage(lithoImage):
        pass

    @staticmethod
    def PlotResistContour(lithoImage, nLevel):
        pass

    @staticmethod
    def PlotFilmStack(filmStack):
        for iARC in range(len(filmStack.Layers)):
            pass
