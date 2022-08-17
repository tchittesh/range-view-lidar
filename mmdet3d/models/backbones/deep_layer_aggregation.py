import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES


class BasicBlock(nn.Module):
    '''Basic Resnet Block
    Conv2d
    BN
    Relu
    Conv2d
    BN
    Relu

    with residual connection between input and output
    '''

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     stride=1, padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.project is not None:
            residual = self.project(residual)
        out += residual
        out = self.relu(out)

        return out


class Deconv(nn.Module):
    '''
    Deconvolution Layer for upsampling
    TransposeConv2d
    BN
    Relu
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Feature_Aggregator(nn.Module):
    '''
    Feature Aggregator Module described in the LaserNet paper
    '''

    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.deconv = Deconv(in_channels_2, out_channels)
        self.block_1 = BasicBlock(in_channels_1+in_channels_2, out_channels)
        self.block_2 = BasicBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x2 = self.deconv(x2)
        x1 = torch.cat([x1, x2], 1)
        x1 = self.block_1(x1)
        x1 = self.block_2(x1)
        return x1


class DownSample(nn.Module):
    '''
    DownSample module using Conv2d with stride > 1
    Conv2d(stride>1)
    BN
    Relu
    Conv2d
    BN
    Relu
    '''

    def __init__(self, in_channels, out_channels, stride=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                 stride=2, padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.project(residual)
        out += residual
        out = self.relu(out)
        return out


class Feature_Extractor(nn.Module):
    '''
    Feature Extrator module described in LaserNet paper
    DownSample input if not 1a
    '''

    def __init__(self, in_channels, out_channels, num_blocks=6,
                 down_sample_input=False):
        super().__init__()
        self.down_sample = None
        self.down_sample_input = down_sample_input
        if down_sample_input:
            self.down_sample = DownSample(in_channels, out_channels)

        blocks_modules = []
        for i in range(num_blocks):
            if i == 0 and not down_sample_input:
                blocks_modules.append(BasicBlock(in_channels, out_channels))
            else:
                blocks_modules.append(BasicBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks_modules)

    def forward(self, x):
        if self.down_sample_input:
            x = self.down_sample(x)
        x = self.blocks(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, channels_num, dim_change=True,
                 custom_stride=(2, 2)):
        super().__init__()
        self.in_channels = in_channels
        self.channels_num = channels_num
        self.dim_change = dim_change
        self.resUnit1 = ResUnit(self.in_channels, channels_num=self.channels_num, filter_size=3, dim_change=dim_change, custom_stride=custom_stride)  # noqa: E501
        self.resUnit2 = ResUnit(self.channels_num, channels_num=self.channels_num, filter_size=3, dim_change=False)  # noqa: E501
        self.resUnit3 = ResUnit(self.channels_num, channels_num=self.channels_num, filter_size=3, dim_change=False)  # noqa: E501
        self.resUnit4 = ResUnit(self.channels_num, channels_num=self.channels_num, filter_size=3, dim_change=False)  # noqa: E501
        if self.dim_change:
            self.reshaping_conv = nn.Conv2d(self.in_channels, self.channels_num, 1, stride=custom_stride, padding=0)  # noqa: E501 1 x 1 conv on the residual connection to change dimension of the residue
        else:
            self.reshaping_conv = nn.Conv2d(self.in_channels, self.channels_num, 1, stride=1, padding=0)  # noqa: E501 1 x 1 conv on the residual connection to change dimension of the residue

    def forward(self, x):
        residue = self.reshaping_conv(x)
        x = self.resUnit1(x)
        x = x + residue
        residue = x
        x = self.resUnit2(x)
        x = x + residue
        residue = x
        x = self.resUnit3(x)
        x = x + residue
        residue = x
        x = self.resUnit4(x)
        x = x + residue
        return x


class ResUnit(nn.Module):
    def __init__(self, in_channels, channels_num, filter_size=3,
                 dim_change=False, custom_stride=(2, 2)):
        super().__init__()
        self.stride = 1
        if dim_change:
            self.stride = custom_stride
        self.conv1 = nn.Conv2d(in_channels, channels_num, filter_size,
                               stride=self.stride, padding=1)
        self.conv2 = nn.Conv2d(channels_num, channels_num, filter_size,
                               stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class AuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resBlock1 = ResBlock(in_channels=3, channels_num=16, dim_change=True)  # noqa: E501
        self.resBlock2 = ResBlock(in_channels=16, channels_num=24, dim_change=True, custom_stride=(1, 2))  # noqa: E501
        self.resBlock3 = ResBlock(in_channels=24, channels_num=32, dim_change=True, custom_stride=(1, 2))  # noqa: E501

    def forward(self, x):
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        return x


@BACKBONES.register_module()
class DeepLayerAggregation(BaseModule):

    def __init__(self,
                 use_rgb=False,
                 lidar_in_channels=5,
                 channels=[64, 128, 256]):
        super().__init__()
        self.use_rgb = use_rgb
        if use_rgb:
            self.rgb_cnn = AuxNet()
        self.lidar_cnn = nn.Conv2d(lidar_in_channels, 32,
                                   kernel_size=3, padding=1)
        num_inputs = 64 if self.use_rgb else 32
        self.extract_1a = Feature_Extractor(num_inputs, channels[0])
        self.extract_2a = Feature_Extractor(channels[0], channels[1],
                                            down_sample_input=True)
        self.extract_3a = Feature_Extractor(channels[1], channels[2],
                                            down_sample_input=True)
        self.aggregate_1b = Feature_Aggregator(channels[0], channels[1],
                                               channels[1])
        self.aggregate_1c = Feature_Aggregator(channels[1], channels[2],
                                               channels[2])
        self.aggregate_2b = Feature_Aggregator(channels[1], channels[2],
                                               channels[2])

    def forward(self, lidar_img, img=None):
        x = self.lidar_cnn(lidar_img)
        if self.use_rgb:
            assert img is not None
            rgb_semantics = self.rgb_cnn.forward(img)
            x = torch.cat((rgb_semantics, x), dim=1)

        x_1a = self.extract_1a(x)
        x_2a = self.extract_2a(x_1a)
        x_3a = self.extract_3a(x_2a)
        x_1b = self.aggregate_1b(x_1a, x_2a)
        x_2b = self.aggregate_2b(x_2a, x_3a)
        x_1c = self.aggregate_1c(x_1b, x_2b)

        return [x_1c, x_2b, x_3a]
