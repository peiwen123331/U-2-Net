import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolution block
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# RSU-7
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.stage1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = REBNCONV(mid_ch, mid_ch)

        self.stage7 = REBNCONV(mid_ch, mid_ch)

        self.stage6d = REBNCONV(mid_ch * 2, mid_ch)
        self.stage5d = REBNCONV(mid_ch * 2, mid_ch)
        self.stage4d = REBNCONV(mid_ch * 2, mid_ch)
        self.stage3d = REBNCONV(mid_ch * 2, mid_ch)
        self.stage2d = REBNCONV(mid_ch * 2, mid_ch)
        self.stage1d = REBNCONV(mid_ch * 2, out_ch)

        self.upscore = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.stage1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool4(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool5(hx5)

        hx6 = self.stage6(hx)

        hx7 = self.stage7(hx6)

        hx6d = self.stage6d(torch.cat((hx7, hx6), 1))
        hx6dup = self.upscore(hx6d)

        hx5d = self.stage5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = self.upscore(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upscore(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

# Full U2Net architecture (U²-Net full)
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU7(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU7(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU7(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU7(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU7(512, 256, 512)

        self.stage5d = RSU7(1024, 256, 512)
        self.stage4d = RSU7(1024, 128, 256)
        self.stage3d = RSU7(512, 64, 128)
        self.stage2d = RSU7(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        hx5d = self.stage5d(torch.cat((hx6, hx5), 1))
        hx4d = self.stage4d(torch.cat((hx5d, hx4), 1))
        hx3d = self.stage3d(torch.cat((hx4d, hx3), 1))
        hx2d = self.stage2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.stage1d(torch.cat((hx2d, hx1), 1))

        d1 = self.side1(hx1d)
        return d1,  # return as a tuple
