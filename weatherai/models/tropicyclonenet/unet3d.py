"""3-D U-Net that rolls the 8 observed 500 hPa geopotential frames forward (official ``Unet3D_merge_tiny``; names match the checkpoint)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    """(conv3 -> BN -> ReLU) x 2 + 1x1x1 residual conv."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(cin, cout, 3, 1, 1), nn.BatchNorm3d(cout), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(cout, cout, 3, 1, 1), nn.BatchNorm3d(cout), nn.ReLU())
        self.residual = nn.Conv3d(cin, cout, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class Down(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(k, k), Conv3dBlock(cin, cout))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, c1, c2, cout, k):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose3d(c1, c1, k, k), nn.ReLU())
        self.conv = Conv3dBlock(c1 + c2, cout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        d = [x2.shape[i] - x1.shape[i] for i in (2, 3, 4)]            # (depth, height, width); sizes here always match
        x1 = F.pad(x1, [d[2] // 2, d[2] - d[2] // 2, d[1] // 2, d[1] - d[1] // 2, d[0] // 2, d[0] - d[0] // 2])
        return self.conv(torch.cat([x2, x1], 1))


class OutConv(nn.Module):
    """Bring the three coarse decoder levels to full resolution, concatenate with the finest along depth, and collapse depth with an 18-tap conv."""

    def __init__(self, chans=(64, 32, 16, 16), cout=1, depth_kernel=18):
        super().__init__()
        n = len(chans)
        ups = []
        for i, c in enumerate(chans[:-1]):
            s = 2 ** (n - 1 - i)
            ups.append(nn.Sequential(nn.ConvTranspose3d(c, c, (1, s, s), (1, s, s)), nn.ReLU(),
                                     nn.Conv3d(c, chans[-1], 3, 1, 1), nn.BatchNorm3d(chans[-1]), nn.ReLU()))
        self.up_list = nn.ModuleList(ups)
        self.conv = nn.Sequential(nn.Conv3d(chans[-1], cout, (depth_kernel, 1, 1)), nn.BatchNorm3d(cout), nn.ReLU(), nn.Conv3d(cout, cout, 1))

    def forward(self, xs):
        x6, x7, x8, x9 = xs
        x = torch.cat([self.up_list[0](x6), self.up_list[1](x7), self.up_list[2](x8), x9], dim=2)
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(self, cin=1, cout=1):
        super().__init__()
        self.inc = Conv3dBlock(cin, 16)
        self.down1, self.down2 = Down(16, 32, (1, 2, 2)), Down(32, 64, (1, 2, 2))
        self.down3, self.down4 = Down(64, 128, (2, 2, 2)), Down(128, 128, (2, 2, 2))
        self.up1, self.up2 = Up(128, 128, 64, (2, 2, 2)), Up(64, 64, 32, (2, 2, 2))
        self.up3, self.up4 = Up(32, 32, 16, (1, 2, 2)), Up(16, 16, 16, (1, 2, 2))
        self.outc = OutConv((64, 32, 16, 16), cout)

    def forward(self, x):                                           # x: (B, 1, 8, 64, 64) -> (B, 1, 11, 64, 64)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        return self.outc([x6, x7, x8, x9])
