""" Estimate 3D Hand Pose through binary Mask
Author: Wayne Lee
"""
import torch.nn as nn

from model.incept_resnet import incept_resnet, ResnetkBasicBlock, PullOut8
from model.DPN import DPN
from model.utils import SlimConv2d, ConvMaxpool, SlimConv2dTranspose

## TODO: (hourglass, Resnetkblock) in & out channel


class Hourglass(nn.module):
    def __init__(self, in_ch, out_ch, ntimes):
        super(Hourglass).__init__()

        self.resnet_k1 = ResnetkBasicBlock()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.resnet_k2 = ResnetkBasicBlock()
        self.conv1 = SlimConv2d(, out_ch*2, kernel_size=1, stride=1)
        self.conv2 = SlimConv2d(, out_ch, kernel_size=1, stride=1)
        self.resnet_k3 = ResnetkBasicBlock()
        self.conv_transpose = SlimConv2dTranspose(, out_ch, kernel_size=3, stride=2)

    def forward(self, x):

        upper0 = self.resnet_k1(x)

        lower0 = self.max_pool1(x)
        lower0 = self.resnet_k2(lower0)
        lower0 = self.conv1(lower0)

        if ntimes > 1:
            lower1 = Hourglass( ?, ?, n-1)
        else:
            lower1 = lower0

        lower1 = self.conv2(lower1)

        lower2 = self.resnet_k3(lower1)
        upper1 = self.conv_transpose(lower2)

        return upper0 + upper1


class RPN(nn.module):
    ### Residual Module ###

    def __init__(self, num_feature, hg_repeat=2, num_joints=21):
        super(RPN)__init__()

        self.conv1 = SlimConv2d(3, 8, kernel_size=3)
        self.conv_maxpool1 = ConvMaxpool(8, 16)
        self.conv_maxpool2 = ConvMaxpool(16, 32)
        self.resnet_k1 = ResnetkBasicBlock(32, 32)
        self.conv2 = SlimConv2d(32, num_feature, kernel_size=1)

        self.hg_net = HGNet()
        self.resnet_k2 = ResnetkBasicBlock()

        self.conv3 = nn.conv2d(?, num_joints*3, kernel_size=1)
        self.conv4 = SlimConv2d(num_joints*3, num_feature, kernel_size=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_maxpool3 = ConvMaxpool3(64, 128)
        self.pullout8 = PullOut8(128, 63)

        # incept_resnet.pullout8(out, self.out_dim) --> (?, 63)
        # self.out_dim =  BxJx3, where B: batch size, J: # of joints

    def forward(self, x):
        # Input: (?, 128, 128, 3)

        stg128 = self.conv1(x)  # (?, 128, 128, 8)
        stg64 = self.conv_maxpool1(stg128)  # (? , 64, 64, 16)
        stg32 = self.conv_maxpool2(stg64)  # (?, 32, 32, 32), scope = 'stage64_image'

        #scope = 'stage32_pre'
        stg32 = self.resnet_k1(stg32)
        out = self.conv2(stg32) #(?, 32, 32, 64)

        for hg in range(hg_repeat):

            ## TODO: we might replace 'hourglass' to other latest framework
            branch0 = self.hg_net(out, 2)
            branch0 = self.resnet_k2(branch0)

            # Multiply bt 3 here, styled Map becomes 63
            heat_maps = self.conv3(branch0)  # (? 32, 32, 63)
            branch1 = self.conv4(heat_maps)

            out += branch0 + branch1

        ## check max pool only?
        #net = incept_resnet.conv_maxpool(net, scope=sc)

        out = self.max_pool2d(out)  # (?, 16, 16, 64)
        out = self.conv_maxpool3(out) # (?, 8, 8, 128)
        out = incept_resnet.pullout8(out) # (?, 63)

        return out

class SilhouetteNet(nn.moduel):
    """
    End-to-end 3D hand pose estimation from a single binary mask
    This class use clean_depth (128, 128, 1), clean_binary(128, 128, 1)
    Plus Multiview data (128, 128, 3)
    'MV' stands for Multi-View

    ### Size of heatmap ###
    # num_feature = 32
    # mv = multi-view = 3

    """

    def __init__(self, mv=3, num_feature=64, hg_repeat=2, is_rgb=False):
        super(SilhouetteNet, self).__init__()

        self.dpn = DPN()
        self.conv1 = SlimConv2d(32, 21*3, stride=1)
        # Q: mp1, mp2, ... or if the params are the same, use only one mp
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2) # max pool padding
        self.conv2 = SlimConv2d(21*3, 21*3, kernel_size=3, stride=1)
        # Check slim.max_pool2d
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = SlimConv2d(128, 21*3, kernel_size=3, stride=1)
        self.conv4 = SlimConv2d(21*3, 21*3, kernel_size=3, stride=1)

        self.rpn = RPN()

    def forward(self, x):

        d1, d2, d3, content_code = self.dpn(x)

        br0 = self.conv1(d3)
        br0 = self.max_pool2d(br0) # (?, 64, 64, 63)
        br1 = self.conv2(d2) # (?, 64, 64, 63), scope = 'hmap64'
        out = br0 + br1
        out = self.max_pool2(out) #scope = 'hmap32', (32, 32, 63)
        br2 = self.conv3(d1) # (?, 32, 32, 63) scope = 'mv_hmap32'
        out = out + br2
        guidance = self.conv4(out) # (?, 32, 32, 63)
        pose = self.rpn(x)

        return guidance, pose


def SilhouetteNet(pretrained=False, **kwargs):
    """Constructs a SilhouetteNet model.
    """
    model = DilatedResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

if __name__ == '__main__':
    print(SilhouetteNet())



