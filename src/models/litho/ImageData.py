
from dataclasses import dataclass
import torch


@dataclass
class ImageData:
    ImageType: str = None  # 1d / 2d
    SimulationType: str = None  # aerial / resisit / Latent / PEB
    Intensity: torch.tensor = None
    ImageX: torch.tensor = None
    ImageY: torch.tensor = None
    ImageZ: torch.tensor = None


if __name__ == '__main__':
    i = ImageData()
    print(i.__dict__)
