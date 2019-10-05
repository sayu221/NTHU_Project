import torch.nn as nn
import torch
from model.utils import SlimConv2d, SlimConv2dTranspose

class DPN(nn.module):

    """
    Generate guidance map

    """
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(fpn_guidance).__init__()

        self.down1 = SlimConv2d(in_ch, 32, kernel_size, stride=1)
        self.down2 = SlimConv2d(32, 32, kernel_size, stride=1)
        self.down3 = SlimConv2d(32, 64, kernel_size, stride=2)
        self.down4 = SlimConv2d(64, 128, kernel_size, stride=2)

        # Transpose do not contains padding
        self.up1 = SlimConv2dTranspose(128, 128, kernel_size, stride=2)
        self._p3 = SlimConv2d(64, 128, kernel_size, stride=1)

        self.up2 = SlimConv2dTranspose(128, 64, kernel_size, stride=2)
        self._p2 = SlimConv2d(32, 64, kernel_size, stride=1)

        self.up3 = SlimConv2dTranspose(64, 32, kernel_size, stride=2)
        self._p1 = SlimConv2d(32, 32, kernel_size, stride=1)

        self.pad = nn.ReflectionPad2d([[0, 0], [3, 3], [3, 3], [0, 0]])
        self.conv = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0)
        self.sigmoid = nn.sigmoid()

    def forward(self, x):

        # (p1, p2, p3, content_code) = (128, 64, 32, 16) --> Resolution

        p1 = self.down1(x)
        p2 = self.down2(p1)
        p3 = self.down3(p2)
        content_code = self.down4(p3)
        # (?, 16, 16, 128)

        d1 = self.up1(content_code)
        r3 = self._p3(p3)
        d1 += r3

        d2 = self.up2(d1)
        r2 = self._p2(p2)
        d2 += r2

        d3 = self.up3(d2)
        r1 = self._p1(p1)
        d3 += r1

        d4 = self.pad(d3)
        fake_depth = self.sigmoid(d4)

        return d1, d2, d3, fake_depth



