import torch
import torch.nn as nn

class SlimBatchNorm2d(nn.Sequential):
    def __init__(self, in_ch, bn_epsilon=0.001):
        super(SlimBatchNorm2d, self).__init__(
                nn.BatchNorm2d(in_ch, eps=bn_epsilon)
        )


class SlimConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(SlimConv2d, self).__init__()
        self.slim_conv = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_ch, out_ch, kernel_size, stride),
                SlimBatchNorm2d(out_ch),
                nn.ReLU()
        )

    def forward(self, x):
        return self.slim_conv(x)


class SlimConv2dTranspose(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=1):
        super(SlimConv2dTranspose, self).__init__()
        self.slim_conv2d_transpose = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride),
                SlimBatchNorm2d(out_ch),
                nn.ReLU()
        )

    def forward(self, x):
        return self.slim_conv2d_transpose(x)


class SlimFullyConnected(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(SlimFullyConnected, self).__init__()
        self.slim_fully_connected = nn.Sequential(
                nn.Linear(in_feature, out_feature),
                nn.BatchNorm1d(out_feature),
                nn.ReLU()
        )

    def forward(self, x):
        return self.slim_fully_connected(x)


class ConvMaxpool(nn.Module):
    """ simple conv + max_pool """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(ConvMaxpool, self).__init__()
        self.conv = SlimConv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.max_pool2d(self.conv(x)) 


#define parameters (in_ch, out_ch, ...)
def PullOut8(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(PullOut8, self).__init__()
        self.conv_maxpool1 = ConvMaxpool(in_ch, out_ch)
        self.conv_maxpool2 = ConvMaxpool(in_ch, out_ch)
        self.conv_maxpool3 = ConvMaxpool(in_ch, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear()

    def forward(self, x):
    """ supposed to work best with 8x8 input """
        x = self.conv_maxpool1(x) # (?, 8, 8, 128)
        x = self.conv_maxpool2(x) # (? ,4, 4, 256)
        x = self.conv_maxpool3(x) # (?, 2, 2, 512)
        x = self.conv(x)
        x = self.dropout(x.view[:,-1])
        x = self.fc(x) # fc_num = 512*2 = 1024
        
        x = self.conv1(x)
        x = self.conv2(x)

        for _ in range(num):
            x = self.conv(x)

        return x
