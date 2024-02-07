import torch
import sys
import math


class ProjectionObjective:
    def __init__(self):
        self.Aberration_Zernike = torch.zeros(37)
        self.Aberration_Polarization = None
        self.PupilFilter = {}  # type : 'none', 'gaussian'
        self.Reduction = 0.25
        self.NA = 1.35  # 0.65-0.93 for dry mode, 1.0-1.35 for immersion mode
        self.LensType = 'Immersion'  # Immersion / Dry
        self.Index_ImmersionLiquid = 1.44   # Air => 1.0, Water => 1.44
        self.PupilFilter['Type'] = 'none'
        self.PupilFilter['Parameters'] = 0

    def CalculateAberration(self, rho, theta, Orientation):
        # Rotate projection objective
        if (abs(Orientation) > sys.float_info.epsilon):
            Coefficients = ProjectionObjective.RoateAngle(
                self.Aberration_Zernike,
                Orientation)
        else:
            Coefficients = self.Aberration_Zernike

        # Calculate aberrations
        aberration = Coefficients[0] * torch.ones(theta.size())\
            + Coefficients[1] * torch.mul(rho, torch.cos(theta))\
            + Coefficients[2] * torch.mul(rho, torch.sin(theta))\
            + Coefficients[3] * (2*rho.pow(2)-1)\
            + Coefficients[4] * torch.mul(rho.pow(2), torch.cos(2*theta))\
            + Coefficients[5] * torch.mul(rho.pow(2), torch.sin(2*theta))\
            + Coefficients[6] * torch.mul(3*rho.pow(3)-2*rho,
                                          torch.cos(theta))\
            + Coefficients[7] * torch.mul(3*rho.pow(3)-2*rho,
                                          torch.sin(theta))\
            + Coefficients[8] * (6*rho.pow(4)-6*rho.pow(2)+1)\
            + Coefficients[9] * torch.mul(rho.pow(3), torch.cos(3*theta))\
            + Coefficients[10] * torch.mul(rho.pow(3), torch.sin(3*theta))\
            + Coefficients[11] * torch.mul(4*rho.pow(4)-3*rho.pow(2),
                                           torch.cos(2*theta))\
            + Coefficients[12] * torch.mul(4*rho.pow(4)-3*rho.pow(2),
                                           torch.sin(2*theta))\
            + Coefficients[13] * torch.mul(10*rho.pow(5)-12*rho.pow(3)+3*rho,
                                           torch.cos(theta))\
            + Coefficients[14] * torch.mul(10*rho.pow(5)-12*rho.pow(3)+3*rho,
                                           torch.sin(theta))\
            + Coefficients[15] * (20*rho.pow(6)-30*rho.pow(4)+12*rho.pow(2)-1)\
            + Coefficients[16] * torch.mul(rho.pow(4), torch.cos(4*theta))\
            + Coefficients[17] * torch.mul(rho.pow(4), torch.sin(4*theta))\
            + Coefficients[18] * torch.mul(5*rho.pow(5)-4*rho.pow(3),
                                           torch.cos(3*theta))\
            + Coefficients[19] * torch.mul(5*rho.pow(5)-4*rho.pow(3),
                                           torch.sin(3*theta))\
            + Coefficients[20] * torch.mul(15*rho.pow(6)-20*rho.pow(4)
                                           + 6*rho.pow(2),
                                           torch.cos(2*theta))\
            + Coefficients[21] * torch.mul(15*rho.pow(6)-20*rho.pow(4)
                                           + 6*rho.pow(2),
                                           torch.sin(2*theta))\
            + Coefficients[22] * torch.mul(35*rho.pow(7)-60*rho.pow(5)
                                           + 30*rho.pow(3)-4*rho,
                                           torch.cos(theta))\
            + Coefficients[23] * torch.mul(35*rho.pow(7)-60*rho.pow(5)
                                           + 30*rho.pow(3)-4*rho,
                                           torch.sin(theta))\
            + Coefficients[24] * (70*rho.pow(8)-140*rho.pow(6)
                                  + 90*rho.pow(4)-20*rho.pow(2)+1)\
            + Coefficients[25] * torch.mul(rho.pow(5), torch.cos(5*theta))\
            + Coefficients[26] * torch.mul(rho.pow(5), torch.sin(5*theta))\
            + Coefficients[27] * torch.mul(6*rho.pow(6)-5*rho.pow(4),
                                           torch.cos(4*theta))\
            + Coefficients[28] * torch.mul(6*rho.pow(6)-5*rho.pow(4),
                                           torch.sin(4*theta))\
            + Coefficients[29] * torch.mul(21*rho.pow(7)-30*rho.pow(5)
                                           + 10*rho.pow(3),
                                           torch.cos(3*theta))\
            + Coefficients[30] * torch.mul(21*rho.pow(7)-30*rho.pow(5)
                                           + 10*rho.pow(3),
                                           torch.sin(3*theta))\
            + Coefficients[31] * torch.mul(56*rho.pow(8)-105*rho.pow(6)
                                           + 60*rho.pow(4)-10*rho.pow(2),
                                           torch.cos(2*theta))\
            + Coefficients[32] * torch.mul(56*rho.pow(8)-105*rho.pow(6)
                                           + 60*rho.pow(4)-10*rho.pow(2),
                                           torch.sin(2*theta))\
            + Coefficients[33] * torch.mul(126*rho.pow(9)-280*rho.pow(7)
                                           + 210*rho.pow(5)-60*rho.pow(3)
                                           + 5*rho,
                                           torch.cos(theta))\
            + Coefficients[34] * torch.mul(126*rho.pow(9)-280*rho.pow(7)
                                           + 210*rho.pow(5)-60*rho.pow(3)
                                           + 5*rho,
                                           torch.sin(theta))\
            + Coefficients[35] * (252*rho.pow(10)-630*rho.pow(8)
                                  + 560*rho.pow(6)-210*rho.pow(4)
                                  + 30*rho.pow(2)-1)\
            + Coefficients[36] * (924*rho.pow(12)-2772*rho.pow(10)
                                  + 3150*rho.pow(8)-1680*rho.pow(6)
                                  + 420*rho.pow(4)-42*rho.pow(2)+1)
        return aberration

    def CalculateAberrationFast(self, rho, theta, Orientation):
        # Perforamance optimized version
        # radial value
        # rValue = ['r2', 'r3', 'r4', 'r5', 'r6',
        #           'r7', 'r8', 'r9', 'r10', 'r12']
        # for i in range(len(rValue)):
        #     exec("%s=%s"%(rValue[i],rho.pow(int(rValue[i][1:]))))
        r2 = rho.pow(2)
        r3 = rho.pow(3)
        r4 = rho.pow(4)
        r5 = rho.pow(5)
        r6 = rho.pow(6)
        r7 = rho.pow(7)
        r8 = rho.pow(8)
        r9 = rho.pow(9)
        r10 = rho.pow(10)
        r12 = rho.pow(12)

        # azimuthal value
        ct = torch.cos(theta)
        st = torch.sin(theta)

        c2t = torch.cos(2*theta)
        s2t = torch.sin(2*theta)

        c3t = torch.cos(3*theta)
        s3t = torch.sin(3*theta)

        c4t = torch.cos(4*theta)
        s4t = torch.sin(4*theta)

        c5t = torch.cos(5*theta)
        s5t = torch.sin(5*theta)

        if (abs(Orientation) > sys.float_info.epsilon):
            Coefficients = ProjectionObjective.RoateAngle(
                self.Aberration_Zernike,
                Orientation)
        else:
            Coefficients = self.Aberration_Zernike

        # calcualte aberration distribution
        aberration = Coefficients[0]
        aberration = aberration + (Coefficients[1] *
                                   torch.mul(rho, ct))
        aberration = aberration + (Coefficients[2] *
                                   torch.mul(rho, st))
        aberration = aberration + (Coefficients[3] *
                                   (2*r2-1))
        aberration = aberration + (Coefficients[4] *
                                   torch.mul(r2, c2t))
        aberration = aberration + (Coefficients[5] *
                                   torch.mul(r2, s2t))
        aberration = aberration + (Coefficients[6] *
                                   torch.mul(3*r3-2*rho, ct))
        aberration = aberration + (Coefficients[7] *
                                   torch.mul(3*r3-2*rho, st))
        aberration = aberration + (Coefficients[8] *
                                   (6*r4-6*r2+1))
        aberration = aberration + (Coefficients[9] *
                                   torch.mul(r3, c3t))
        aberration = aberration + (Coefficients[10] *
                                   torch.mul(r3, s3t))
        aberration = aberration + (Coefficients[11] *
                                   torch.mul(4*r4-3*r2, c2t))
        aberration = aberration + (Coefficients[12] *
                                   torch.mul(4*r4-3*r2, s2t))
        aberration = aberration + (Coefficients[13] *
                                   torch.mul(10*r5-12*r3+3*rho, ct))
        aberration = aberration + (Coefficients[14] *
                                   torch.mul(10*r5-12*r3+3*rho, st))
        aberration = aberration + (Coefficients[15] *
                                   (20*r6-30*r4+12*r2-1))
        aberration = aberration + (Coefficients[16] *
                                   torch.mul(r4, c4t))
        aberration = aberration + (Coefficients[17] *
                                   torch.mul(r4, s4t))
        aberration = aberration + (Coefficients[18] *
                                   torch.mul(5*r5-4*r3, c3t))
        aberration = aberration + (Coefficients[19] *
                                   torch.mul(5*r5-4*r3, s3t))
        aberration = aberration + (Coefficients[20] *
                                   torch.mul(15*r6-20*r4+6*r2, c2t))
        aberration = aberration + (Coefficients[21] *
                                   torch.mul(15*r6-20*r4+6*r2, s2t))
        aberration = aberration + (Coefficients[22] *
                                   torch.mul(35*r7-60*r5+30*r3-4*rho, ct))
        aberration = aberration + (Coefficients[23] *
                                   torch.mul(35*r7-60*r5+30*r3-4*rho, st))
        aberration = aberration + (Coefficients[24] *
                                   (70*r8-140*r6+90*r4-20*r2+1))
        aberration = aberration + (Coefficients[25] *
                                   torch.mul(r5, c5t))
        aberration = aberration + (Coefficients[26] *
                                   torch.mul(r5, s5t))
        aberration = aberration + (Coefficients[27] *
                                   torch.mul(6*r6-5*r4, c4t))
        aberration = aberration + (Coefficients[28] *
                                   torch.mul(6*r6-5*r4, s4t))
        aberration = aberration + (Coefficients[29] *
                                   torch.mul(21*r7-30*r5+10*r3, c3t))
        aberration = aberration + (Coefficients[30] *
                                   torch.mul(21*r7-30*r5+10*r3, s3t))
        aberration = aberration + (Coefficients[31] *
                                   torch.mul(56*r8-105*r6+60*r4-10*r2, c2t))
        aberration = aberration + (Coefficients[32] *
                                   torch.mul(56*r8-105*r6+60*r4-10*r2, s2t))
        aberration = aberration + (Coefficients[33] *
                                   torch.mul(126*r9-280*r7+210*r5
                                             - 60*r3+5*rho,
                                             ct))
        aberration = aberration + (Coefficients[34] *
                                   torch.mul(126*r9-280*r7+210*r5
                                             - 60*r3+5*rho,
                                             st))
        aberration = aberration + (Coefficients[35] *
                                   (252*r10-630*r8+560*r6
                                    - 210*r4+30*r2-1))
        aberration = aberration + (Coefficients[36] *
                                   (924*r12-2772*r10+3150*r8
                                    - 1680*r6+420*r4-42*r2+1))
        return aberration

    def CalculateAberrationEasy(self, rho, theta, Orientation):
        if (abs(Orientation) > sys.float_info.epsilon):
            Coefficients = ProjectionObjective.RoateAngle(
                self.Aberration_Zernike,
                Orientation)
        else:
            Coefficients = self.Aberration_Zernike
        Aberration = 0
        for N in range(6):
            for n in range(N,2*N):
                co_index = N**2 + (n - N) * 2
                m1 = 2 * N - n
                m2 = n - 2 * N 
                Aberration = Aberration \
                           + Coefficients[co_index] * self.Zerniken(n, m1, rho, theta) \
                           + Coefficients[co_index+1] * self.Zerniken(n, m2, rho, theta) 
            Aberration = Aberration + Coefficients[(N+1)**2-1] * self.Zerniken(2*N, 0, rho, theta)
        Aberration = Aberration + Coefficients[36] * self.Zerniken(12, 0 ,rho, theta)
        return Aberration

    def CalculatePupilFilter(self, rho, theta):
        parameters = self.PupilFilter
        if (parameters['Type'] == 'gaussian'):
            # filter = parameters['Type']
            # para = parameters['Parameters']
            pupilFilter = 1  # TODO
        elif (self.PupilFilter['Type'] == 'none'):
            pupilFilter = 1
        else:
            pupilFilter = 1
        return pupilFilter

    @staticmethod
    def Zerniken(n, m, rho, theta):
        Rnm = torch.zeros(rho.shape)
        S = int((n - abs(m)) / 2)
        for s in range(S + 1):
            CR = (
                pow(-1, s)
                * math.factorial(n - s)
                / (
                    math.factorial(s)
                    * math.factorial(-s + int((n + abs(m)) / 2))
                    * math.factorial(-s + int((n - abs(m)) / 2))
                )
            )
            p = CR * pow(rho, n - 2 * s)
            Rnm = Rnm + p
        Rnm[rho > 1.0] = 0
        if m >= 0:
            Zmn = Rnm * torch.cos(abs(m) * theta)
        elif m < 0:
            Zmn = Rnm * torch.sin(abs(m) * theta)
        return Zmn

    @staticmethod
    def RoateAngle(c0, theta):  # Rotate zernike aberration
        # The aberration relationship to the aberration of the 1D mask on the axis
        # for a 1D mask that is not on the axis
        # P.S.
        # 1. There is no change in COS and SIN items
        # 2. The cos term is equal to itself multiplied by cos
        #    plus the corresponding sin-containing term multiplied by sin
        # 3. The sin term is equal to itself multiplied by sin
        #    plus the corresponding cos-containing term multiplied by cos
        mm = torch.tensor([0, 1, 1, 0, 2, 2,
                           1, 1, 0, 3, 3, 2,
                           2, 1, 1, 0, 4, 4,
                           3, 3, 2, 2, 1, 1,
                           0, 5, 5, 4, 4, 3,
                           3, 2, 2, 1, 1, 0, 0])  # m
        tt = torch.tensor([0, 1, -1, 0, 1, -1,
                           1, -1, 0, 1, -1, 1,
                          -1, 1, -1, 0, 1, -1,
                           1, -1, 1, -1, 1, -1,
                           0, 1, -1, 1, -1, 1,
                          -1, 1, -1, 1, -1, 0, 0])
        pp = torch.tensor([0, 2, 1, 3, 5, 4, 7, 6,
                           8, 10, 9, 12, 11, 14,
                           13, 15, 17, 16, 19, 18,
                           21, 20, 23, 22, 24, 26,
                           25, 28, 27, 30, 29, 32,
                           31, 34, 33, 35, 36])

        c1 = torch.zeros(37)
        for ii in range(37):
            c1[ii] = c1[ii] + c0[ii] * torch.cos(mm[ii] * theta)
            c1[pp[ii]] = c1[pp[ii]] - tt[ii] * c0[ii]\
                * torch.sin(mm[ii]*theta)
        return c1


if __name__ == '__main__':
    po = ProjectionObjective()
    rho = torch.tensor([0.15, 0.35, 0.55, 0.75, 0.95])
    theta = math.pi/2 * \
        torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    po.Aberration_Zernike = torch.ones(37)
    print(po.CalculateAberration(rho, theta, 0))
    print(po.CalculateAberrationFast(rho, theta, 0))
    # for i in range(37):
    #     print(po.Zerniken(i, rho, theta))
    print(po.CalculateAberrationEasy(rho, theta, 0))
