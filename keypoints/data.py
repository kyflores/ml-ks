from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import torchvision.transforms.v2 as tv2
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import spectral
spectral.settings.envi_support_nonlowercase_params = True

class RawScan(tud.Dataset):
    def __init__(self, path: Path, wsz: int, stepsz: tuple[int, int]):
        super().__init__()
        self.cube: spectral.io.bsqfile.BsqFile = envi.open(path)
        self.crop = tv2.CenterCrop(wsz)
        self.min_step, self.max_step = stepsz

        # Profile mean/std of the cube from a subset of frames
        stats = []
        for x in random.choices(range(self.cube.nbands), k=min(250, self.cube.nbands)):
            stats.append(torch.from_numpy(self.cube.read_band(x)).float())

        stats = self.crop(torch.stack(stats))
        self.std, self.mean = torch.std_mean(stats)

        self.proc = nn.Sequential(
            tv2.ToDtype(dtype=torch.float32, scale=False),
            tv2.Normalize([self.mean.item()], [self.std.item()]),
            tv2.RandomHorizontalFlip(0.5),
            tv2.RandomVerticalFlip(0.5),
            tv2.RandomAffine(
                degrees=[-5, 5],
                # translate=(0.25, 0.25),
                scale=(0.9, 1.1),
                fill=0.5
            ),
            tv2.RandomResizedCrop((256, 256), antialias=True),
            tv2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
            tv2.GaussianNoise(clip=False)
        )


    def __len__(self) -> int:
        return self.cube.nbands - self.max_step


    def __getitem__(self, index) -> torch.Tensor:
        def load(index):
            frame = self.cube.read_band(index, use_memmap=True)
            frame = torch.from_numpy(frame)
            frame = self.crop(frame)
            # frame = (frame - self.mean) / self.std
            return frame

        s = random.randint(self.min_step, self.max_step)

        frame = torch.stack([load(index), load(index + s)])
        frame = self.proc(frame)

        return frame



if __name__ == '__main__':
    z = RawScan(
        path=Path('/home/kyle/Documents/keypoints/UTK_Airborne/20250228/SPI_20250228173835_[AIRBORNE]_MIDFOV_2x.hdr'),
        wsz=512,
        stepsz=(1, 8)
    )

    f = z[0]
    plt.imshow((f[0]).abs().cpu())
    plt.show()
    plt.imshow((f[1]).abs().cpu())
    plt.show()
