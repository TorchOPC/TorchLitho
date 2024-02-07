class Material:
    name = None
    n = None  # refractive index
    k = None  # extinction coefficient
    wavelength = None
    # any material
    def __init__(self, name = None, n = None, k = None, wavelength = None):
        self.name = name
        self.n = n
        self.k = k
        self.wavelength = wavelength

    # material 'Si'
    @staticmethod
    def Silicon():
        m = Material()
        m.name = 'Si'
        m.n = 0.883143
        m.k = 2.777792
        m.wavelength = 193
        return m

    # material 'SiO2'
    @staticmethod
    def SiliconDioxide():
        m = Material()
        m.name = 'SiO2'
        m.n = 1.563117
        m.k = 0
        m.waveLength = 193
        return m

    # material 'AZ Aquatar ArF Top ARC'
    @staticmethod
    def ARCTop():
        m = Material()
        m.name = 'AZ Aquatar ArF Top ARC'
        m.n = 1.513282
        m.k = 0.004249
        m.waveLength = 193
        return m

    # material 'Brewer DUV 42C'
    @staticmethod
    def ARCBottom():
        m = Material()
        m.name = 'Brewer DUV 42C'
        m.n = 1.48
        m.k = 0.41
        m.waveLength = 193
        return m


if __name__ == "__main__":
    Si = Material.Silicon()
    print(Si.__dict__)
