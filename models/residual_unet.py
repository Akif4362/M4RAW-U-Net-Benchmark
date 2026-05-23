import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class ResCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.Conv = conv_block(ch_in, ch_out)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv_1x1(x)
        x = self.Conv(x)
        return x + x1


class ResU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ResCNN1 = ResCNN_block(ch_in=img_ch, ch_out=64)
        self.ResCNN2 = ResCNN_block(ch_in=64, ch_out=128)
        self.ResCNN3 = ResCNN_block(ch_in=128, ch_out=256)
        self.ResCNN4 = ResCNN_block(ch_in=256, ch_out=512)
        self.ResCNN5 = ResCNN_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_ResCNN5 = ResCNN_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_ResCNN4 = ResCNN_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_ResCNN3 = ResCNN_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_ResCNN2 = ResCNN_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.ResCNN1(x)
        x2 = self.Maxpool(x1); x2 = self.ResCNN2(x2)
        x3 = self.Maxpool(x2); x3 = self.ResCNN3(x3)
        x4 = self.Maxpool(x3); x4 = self.ResCNN4(x4)
        x5 = self.Maxpool(x4); x5 = self.ResCNN5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_ResCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_ResCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_ResCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_ResCNN2(d2)

        return self.Conv_1x1(d2)
