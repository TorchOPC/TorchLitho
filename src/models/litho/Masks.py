"""TODO:remove the IMAGE_WH constant."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from gdsii.library import Library

IMAGE_WH = 2048


class Mask:
    """Binary Mask.

    Args:
        x/ymax: for the computing area
        x/y_gridsize: the simulated size of the area. Different value are supported. 2nm
        CD: used for method poly2mask, 45nm

    .. plot::
       :include-source:

        import matplotlib.pyplot as plt

        from litho.config import PATH
        from litho.mask import Mask

        m = Mask()
        m.x_range = [-300.0, 300.0]
        m.y_range = [-300.0, 300.0]
        m.x_gridsize = 10
        m.y_gridsize = 10
        m.openGDS(PATH.gdsdir / "AND2_X4.gds", 10)
        m.maskfft()
        m.smooth()

        plt.imshow(
            m.data,
            extent=(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1]),
            cmap="hot",
            interpolation="none",
        )
        plt.figure()
        plt.imshow(
            m.sdata,
            extent=(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1]),
            cmap="hot",
            interpolation="none",
        )
        plt.show()
    """

    def __init__(
        self,
        gds_path: str,
        layername: int = 11,
        pixels_per_um: int = 100,
        xmax=1024,
        ymax=1024,
        x_gridsize=1,
        y_gridsize=1,
        CD=45,
    ):
        self.x_range = [-xmax, xmax]  # nm
        self.y_range = [-ymax, ymax]
        self.x_gridsize = x_gridsize  # nm
        self.y_gridsize = y_gridsize
        self.mask_groups = []
        self.CD = CD
        self.gds_path = gds_path
        self.layername = layername
        self.pixels_per_um = pixels_per_um

        """
        Process calculation
        """
        self.openGDS()
        self.maskfft()

    def poly2mask(self):
        """Get Pixel-based Mask Image from Polygon Data The Poylgon Data Form are sensitive Similar
        to poly2mask in Matlab."""
        self.x_gridnum = int((self.x_range[1] - self.x_range[0]) / self.x_gridsize)
        self.y_gridnum = int((self.y_range[1] - self.y_range[0]) / self.y_gridsize)
        img = Image.new("L", (self.x_gridnum, self.y_gridnum), 0)

        self.perimeter = 0.0
        for ii in self.polygons:
            pp = np.array(ii) * self.CD  # polygon
            polygonlen = len(pp)
            self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))
            pp[:, 0] = (pp[:, 0] - self.x_range[0]) / self.x_gridsize
            pp[:, 1] = (pp[:, 1] - self.y_range[0]) / self.y_gridsize
            vetex_list = list(pp)
            polygon = [tuple(y) for y in vetex_list]
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        self.data = torch.from_numpy(np.array(img))
        # self.data = np.float64(self.data)
        self.spat_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)

        self.freq_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)

    def openGDS(self):
        gdsdir = self.gds_path
        layername = self.layername
        pixels_per_um = self.pixels_per_um

        with open(gdsdir, "rb") as stream:
            lib = Library.load(stream)

        a = lib.pop(0)
        b = []
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        for ii in range(0, len(a)):
            if a[ii].layer == layername:
                # if hasattr(a[ii],'data_type'):
                if len(a[ii].xy) > 1:
                    aa = np.array(a[ii].xy) / 1000 * pixels_per_um
                    b.append(aa)
                    xmin.append(min([k for k, v in aa]))
                    xmax.append(max([k for k, v in aa]))
                    ymin.append(min([v for k, v in aa]))
                    ymax.append(max([v for k, v in aa]))
        self.polylist = b

        xmin = min(xmin)
        xmax = max(xmax)
        ymin = min(ymin)
        ymax = max(ymax)

        center_x = (xmax - xmin) // 2
        center_y = (ymax - ymin) // 2

        cpoints = []

        cx_r = np.arange(center_x, xmax, IMAGE_WH // 2)
        cx_l = -np.arange(-center_x, -xmin, IMAGE_WH // 2)
        cxs = np.hstack((cx_l, cx_r))

        cy_u = np.arange(center_y, ymax, IMAGE_WH // 2)
        cy_d = -np.arange(-center_y, -ymin, IMAGE_WH // 2)
        cys = np.hstack((cy_d, cy_u))
        # cys = np.arange(ymin, ymax - IMAGE_WH // 2, IMAGE_WH // 2)

        for x in cxs:
            for y in cys:
                cpoints.append((x, y))

        cpoints = list(set(cpoints))
        # center_x = (xmax - xmin) // 2
        # center_y = (ymax - ymin) // 2
        # xmin = center_x - (IMAGE_WH // 2)
        # ymin = center_y - (IMAGE_WH // 2)
        # xmax = xmin + IMAGE_WH
        # ymax = ymin + IMAGE_WH

        # spoints.append((xmin, ymin))

        # print(spoints)

        for cc in cpoints:
            self.xmin = cc[0] - (IMAGE_WH // 2)
            self.xmax = self.xmin + IMAGE_WH
            self.ymin = cc[1] - (IMAGE_WH // 2)
            self.ymax = self.ymin + IMAGE_WH
            self.x_range = [self.xmin, self.xmax]
            self.y_range = [self.ymin, self.ymax]

            self.x_gridnum = int((self.xmax - self.xmin) / self.x_gridsize)
            self.y_gridnum = int((self.ymax - self.ymin) / self.y_gridsize)
            img = Image.new("L", (self.x_gridnum, self.y_gridnum), 0)

            self.perimeter = 0.0
            for ii in self.polylist:
                pp = np.array(ii)  # polygon
                polygonlen = len(pp)
                self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))

                pp[:, 0] = (pp[:, 0] - self.xmin) / self.x_gridsize
                pp[:, 1] = (pp[:, 1] - self.ymin) / self.y_gridsize
                vetex_list = list(pp)
                polygon = [tuple(y) for y in vetex_list]
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

                self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))

            self.mask_groups.append(torch.from_numpy(np.array(img)))

        self.data = self.mask_groups[0]
        # Fourier transform pair, pyfftw syntax
        self.spat_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)
        self.freq_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)

    # use the fftw packages
    def maskfft(self):
        self.spat_part[:] = torch.fft.ifftshift(self.data)
        self.freq_part = torch.fft.fftn(self.spat_part)
        self.fdata = torch.fft.fftshift(self.freq_part)

    def maskfftold(self):
        self.fdata = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(self.data)))

    def smooth(self):
        xx = torch.linspace(-1, 1, 21)
        X, Y = torch.meshgrid(xx, xx, indexing="xy")
        R = X**2 + Y**2
        # change G to 4-D to use F.conv2d
        G = torch.exp(-10 * R)
        G = G.view(1, 1, *G.shape)
        # print(G.shape)
        # print(self.data.shape)
        D = F.conv2d(0.9 * self.data.unsqueeze(0) + 0.05, G, padding="same") / torch.sum(G)
        self.sdata = D.squeeze().to(torch.float64)
        # print(self.sdata.shape)
