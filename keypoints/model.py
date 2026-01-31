import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as tvm


def get_2d_gaussian(shape, centers, stds):
    B, K, _ = centers.shape
    H, W = shape
    device = centers.device

    # Generate normalized grid coordinates
    y_coords = torch.linspace(-1.0, 1.0, H, device=device)
    x_coords = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_y = grid_y[None, None].expand(B, K, -1, -1)
    grid_x = grid_x[None, None].expand(B, K, -1, -1)

    # Compute normalized distance from center for each dimension
    grid_y = grid_y - centers[..., [0]].unsqueeze(-1)
    grid_x = grid_x - centers[..., [1]].unsqueeze(-1)


    dy_std = grid_y / stds[..., None, None, None]
    dx_std = grid_x / stds[..., None, None, None]

    # Compute Gaussian kernel
    exponent = -0.5 * (dy_std ** 2 + dx_std ** 2)
    gaussian = torch.exp(exponent)

    return gaussian


class HeatmapToGaussian(nn.Module):
    def __init__(self):
        super().__init__()

    # x has shape [B, C, H, W]
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

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

        return torch.stack((hev, wev), dim=-1)

# Conv Relu Batchnorm with optional spatial downsampling
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

class ResDown(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ksz: int = 3,
        ):
        super().__init__()
        self.conv0 = ConvActBN(in_channels, in_channels * 2, ksz=ksz)
        self.conv1 = ConvActBN(in_channels * 2, in_channels * 2, ksz=ksz)

        self.up_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=(1, 1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        xid = x
        xid = self.up_proj(xid)

        x = self.conv0(x)
        x = self.conv1(x)
        x = x + xid
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

class GeomEncoder(nn.Module):
    def __init__(self, num_pts, ch=32):
        super().__init__()

        # self.conv0 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=ch,
        #     kernel_size=(7, 7),
        #     padding=(3, 3),
        #     padding_mode="replicate"
        # )

        # self.res1 = ResDown(in_channels=ch    , ksz=3)
        # self.res2 = ResDown(in_channels=ch * 2, ksz=3)
        # self.res3 = ResDown(in_channels=ch * 4, ksz=3)
        # self.res4 = ResDown(in_channels=ch * 8, ksz=3)

        # Use pretrained down model
        resnet = tvm.resnet18(tvm.ResNet18_Weights.DEFAULT)
        self.resnet_pt = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            # resnet.layer4,
        )

        self.out = nn.Conv2d(
            in_channels=256,
            out_channels=num_pts,
            kernel_size=(1, 1)
        )

        self.heatmap = HeatmapToGaussian()

    def forward(self, x):
        b = x.shape[0]
        x = x.expand(-1, 3, -1, -1)
        x = self.resnet_pt(x)
        x = self.out(x)
        centers = self.heatmap(x)
        gaussians =  get_2d_gaussian((16, 16), centers, 0.1 * torch.ones(b, device=x.device))

        return x, gaussians


class Generator(nn.Module):
    def __init__(self, num_pts, ch=32):
        super().__init__()

        self.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=ch,
            kernel_size=(7, 7),
            padding=(3, 3),
            padding_mode="replicate"
        )

        # self.res1 = ResDown(in_channels=ch    , ksz=3)
        # self.res2 = ResDown(in_channels=ch * 2, ksz=3)
        # self.res3 = ResDown(in_channels=ch * 4, ksz=3)
        # self.res4 = ResDown(in_channels=ch * 8, ksz=3)

        resnet = tvm.resnet18(tvm.ResNet18_Weights.DEFAULT)
        self.resnet_pt = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            # resnet.layer4,
        )


        self.res5 = ResUp(in_channels= num_pts + ch * 8, out_channels=ch * 4, ksz=3)
        self.res6 = ResUp(in_channels=ch * 4 , out_channels=ch * 2, ksz=3)
        self.res7 = ResUp(in_channels=ch * 2 , out_channels=ch * 1, ksz=3)
        self.res8 = ResUp(in_channels=ch * 1 , out_channels=ch * 1, ksz=3)

        self.out = nn.Conv2d(
            in_channels=ch,
            out_channels=1,
            kernel_size=(1, 1)
        )


    def forward(self, x, heatmaps):
        b = x.shape[0]

        #x = self.conv0(x)

        #x = self.res1(x)
        #x = self.res2(x)
        #x = self.res3(x)
        #x = self.res4(x)

        x = x.expand(-1, 3, -1, -1)
        x = self.resnet_pt(x)


        x = torch.cat((x, heatmaps), dim=1)

        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)

        x = self.out(x)

        return x

def viz_pts(img, heatmaps):
    assert img.shape[0] == 1
    assert heatmaps.shape[0] == 1

    upx = img.shape[2] // heatmaps.shape[2]
    upy = img.shape[3] // heatmaps.shape[3]

    hm = nn.functional.interpolate(heatmaps, scale_factor=(upy, upx), mode='bilinear')

    am = torch.amax(hm, dim=(-2, -1))

    plt.imshow(img.cpu().squeeze())
    for x in range(heatmaps.shape[1]):
        pt = ((hm[0, x] == am[0, x]).nonzero()).squeeze()
        plt.plot(pt[0], pt[1], 'o')
    plt.show()


if __name__ == '__main__':
    hmg = HeatmapToGaussian()
    z = hmg(torch.randn(32, 10, 16, 16))

    #z = torch.tensor([
    #    [-0.25, -0.25],
    #    [-0.25, -0.25],
    #    [-0.25, -0.25],
    #])

    # y = get_2d_gaussian(
    #     torch.tensor((256, 256)),
    #     z,
    #     torch.tensor((0.5, 2.0, 10))
    # )


    # enc = ConvActBN(32, 3, True)
    # w = enc(torch.randn(100, 32, 16, 16))
    # print(w.shape)

    # dec = DecConvActBN(32, 3, False)
    # w = dec(torch.randn(100, 32, 16, 16))
    # print(w.shape)


    g = GeomEncoder(10)
    fakedata = torch.randn(7, 1, 256, 256)
    heatmaps, gaussians =  g(fakedata)

    gen = Generator(10)
    b = gen(
        x=torch.randn(7, 1, 256, 256),
        heatmaps=gaussians
    )
