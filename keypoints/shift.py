import os
import torch
import torch.nn as nn
import torchvision.models as tvm
import random
import torchvision.transforms.v2 as tv2
import kornia as K
import imageio.v3 as iio
import torch.utils.data as tud
from skimage.filters import window
from tqdm import tqdm

import matplotlib.pyplot as plt

class ShiftDataset(tud.Dataset):
    def __init__(self, img_dir, max_shift, crop_size):
        self.max_shift=max_shift

        self.img_dir = img_dir
        img_names = os.listdir(img_dir)
        self.img_names = [x for x in img_names if (x.endswith(".jpg") or x.endswith(".png"))]

        self.crop_size=crop_size

        self.post = nn.Sequential(
            # tv2.Normalize([self.mean.item()], [self.std.item()]),
            tv2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
            tv2.GaussianNoise(clip=False)
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img = torch.from_numpy(
            iio.imread(
                os.path.join(self.img_dir, self.img_names[index]),
            )
        ).float().mean(dim=-1).unsqueeze(0)

        shift_px_x = random.randint(-self.max_shift, self.max_shift)
        shift_px_y = random.randint(-self.max_shift, self.max_shift)

        shifted = K.geometry.translate(
            img,
            torch.tensor([[shift_px_x, shift_px_y]], dtype=torch.float32),
            mode = "bilinear",
            padding_mode="border"
        )
        sample = torch.cat([img, shifted], dim=0)
        # Cut off dark borders from translation
        sample = sample[..., self.max_shift : -self.max_shift, self.max_shift : -self.max_shift]

        # Take a random crop of the desized size
        _, h, w = sample.shape
        roi_x = random.randint(0, w - self.crop_size)
        roi_y = random.randint(0, h - self.crop_size)
        sample = sample[...,  roi_y : roi_y + self.crop_size, roi_x: roi_x + self.crop_size]

        # Assumes the input was 8 bit.
        sample = sample / 255

        # Add noise and blur
        sample = self.post(sample)

        # Return in Y, X order to match the phase correlation function
        return sample.clone(), torch.tensor([shift_px_y, shift_px_x])

class PhaseCorrLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # x has shape [B, C, H, W]
    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        assert x0.shape == x1.shape
        b, h, w = x0.shape
        win = torch.from_numpy(window("hann", (h, w))).float().to(x0.device)

        s0 = torch.fft.fft2(x0 * win)
        s1 = torch.fft.fft2(x1 * win)

        prod = s0 * s1.conj()
        it = torch.fft.ifft2(prod)

        x = torch.fft.ifftshift(it, dim=(-2, -1)).abs()

        # Separable version
        # Reduce H expected value
        gc_prob = x.mean(dim=-1)
        gc_prob = gc_prob.softmax(dim=-1)
        coord_pt = torch.linspace(-1.0, 1.0, h, device=x.device)
        hev = (gc_prob * coord_pt).sum(dim=-1)

        # Reduce W expected value
        gc_prob = x.mean(dim=-2)
        gc_prob = gc_prob.softmax(dim=-1)
        coord_pt = torch.linspace(-1.0, 1.0, w, device=x.device)
        wev = (gc_prob * coord_pt).sum(dim=-1)

        return -1 * torch.stack((hev, wev), dim=-1)

class ConvActBN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ksz: int = 3,
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (ksz, ksz),
            stride=1,
            padding=ksz//2,
            padding_mode="replicate",
            bias=True
        )

        self.act = nn.ReLU()

        self.bn = nn.BatchNorm2d(
            out_channels
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        return x

class ResUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksz: int = 3,
    ):
        super().__init__()
        self.down_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = ConvActBN(in_channels, out_channels, ksz=ksz)
        self.conv1 = ConvActBN(out_channels, out_channels, ksz=ksz)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, stride=2, kernel_size=(2, 2))

    def forward(self, x):
        x = self.up(x)
        xid = x
        xid = self.down_proj(xid)

        x = self.conv0(x)
        x = self.conv1(x)
        x = x + xid
        return x


class CnnModel(nn.Module):
    def __init__(self,ch=32):
        super().__init__()

        self.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=ch,
            kernel_size=(7, 7),
            padding=(3, 3),
            padding_mode="replicate"
        )

        resnet = tvm.resnet18(tvm.ResNet18_Weights.DEFAULT)
        self.resnet_pt = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )


        self.res5 = ResUp(in_channels=ch * 8, out_channels=ch * 4, ksz=3)
        self.res6 = ResUp(in_channels=ch * 4 , out_channels=ch * 2, ksz=3)
        self.res7 = ResUp(in_channels=ch * 2 , out_channels=ch * 1, ksz=3)
        self.res8 = ResUp(in_channels=ch * 1 , out_channels=ch * 1, ksz=3)

        self.out = nn.Conv2d(
            in_channels=ch,
            out_channels=1,
            kernel_size=(1, 1)
        )


    def forward(self, x):
        b = x.shape[0]
        x = x.expand(-1, 3, -1, -1)
        x = self.resnet_pt(x)

        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)

        x = self.out(x)

        return x


def save_state(name, epoch, model, optim, sched=None):
    data = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched
    }
    torch.save(data, name)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    restore = False
    lr = 2e-3
    epochs = 100

    model = CnnModel(ch=32).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    if restore:
        z = torch.load("checkpoint_25.pth", weights_only=False)
        model.load_state_dict(z["model"])
        optim.load_state_dict(z["optim"])

    pcl = PhaseCorrLoss()
    criterion = torch.nn.MSELoss()

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    dataset = ShiftDataset("./data", 32, 256)
    dataloader = tud.DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)

    losses = []
    for epoch in range(epochs):
        for i, (frames, translations) in enumerate(tqdm(dataloader)):
            frames = frames.to(device)
            p0 = model(frames[:, [0], ...])
            p1 = model(frames[:, [1], ...])

            print(p0.shape)

            ev = pcl(p0.squeeze(1), p1.squeeze(1))
            loss = criterion(ev, translations.to(device=device, dtype=torch.float32))
            losses.append(loss.item())
            loss.backward()
            optim.step()

        if epoch % 5 == 0:
            save_state(f"checkpoint_{epoch}.pth", epoch, model, optim, None)
        # scheduler.step()

        print("Loss for epoch {}: {}".format(epoch,torch.tensor(losses).mean()))



if __name__ == '__main__':
    train()
    # d = ShiftDataset("./data", 32, 1024)
    # pcl = PhaseCorrLoss()
    # s, dxy = d[0]
    # plt.imshow(s[0])
    # plt.show()
    # plt.imshow(s[1])
    # plt.show()
    # s = s.unsqueeze(0)


    # print("S:", s.shape)
    # s.requires_grad_()
    # ev = pcl(s[:, 0, ...], s[:, 1, ...])
    # print(dxy, (ev * 1024).squeeze())

    # loss = torch.mean((dxy/1024 - (ev).squeeze())**2)
    # print("LOSS:",loss)
    # loss.backward()

