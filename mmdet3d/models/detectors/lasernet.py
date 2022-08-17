from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Base3DDetector
from utils import radius_nms_np, kmeans


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


class Deep_Aggregation(nn.Module):
    '''
    Main Deep Aggregation class described as in LaserNet paper
    num_outputs is the number of channels of the output image
    output image has the same width and height as input image
    '''

    def __init__(self, num_inputs, channels, num_outputs):
        super().__init__()
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
        self.conv_1x1 = nn.Conv2d(channels[2], num_outputs,
                                  kernel_size=1, stride=1)

    def forward(self, x):
        x_1a = self.extract_1a(x)
        x_2a = self.extract_2a(x_1a)
        x_3a = self.extract_3a(x_2a)
        x_1b = self.aggregate_1b(x_1a, x_2a)
        x_2b = self.aggregate_2b(x_2a, x_3a)
        x_1c = self.aggregate_1c(x_1b, x_2b)
        out = self.conv_1x1(x_1c)
        return out


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


class Radius_NMS(nn.Module):
    '''
    Radius NMS class used to execute the radius nms taking the LaserNet
    segmentation output as input and returing the reduced boxes
    '''
    def __init__(self, num_classes, min_height, max_height, radius=3):
        super().__init__()
        self.num_classes = num_classes
        self.min_height = min_height
        self.max_height = max_height
        self.radius = radius

    def forward(self, segmented_channels, lidar, verbose=False):
        _, ind = segmented_channels.max(dim=1)
        batch_size = segmented_channels.shape[0]

        start = perf_counter()
        results = []
        for b in range(batch_size):
            class_res = []
            for c in range(1, self.num_classes):
                class_cond = ind[b] == c

                lidar_points_x = lidar[b][0][class_cond].reshape(-1, 1)
                lidar_points_y = lidar[b][1][class_cond].reshape(-1, 1)

                preds = torch.cat(
                    (lidar_points_x, lidar_points_y), dim=1).cuda()
                if preds.shape[0] > 0:
                    res = radius_nms_np(
                        preds.detach().cpu().numpy(), self.radius)
                    class_res.append(preds[res])
                else:
                    class_res.append([])
            results.append(class_res)
        if verbose:
            print(f"SoftNMS time: {(perf_counter()-start)/1e6} ms")

        return results


class LaserNet(Base3DDetector):
    '''
    LaserNet network as described in the original paper, created using
    the Deep Aggregation Network. The fusion between RGB and Lidar used
    is similar to the one in the LaserNet++ paper.

    Complete model including LaserNet and radius nms, a forward pass would take
    Lidar and RGB as input and return the predicted boxes.
    K-means is an option to use to improve the box center predictions.
    '''

    def __init__(self,
                 use_rgb=False,
                 deep_aggregation_num_channels=[64, 64, 128],
                 num_classes=3,
                 min_height=0,
                 max_height=2.5,
                 k_means_iters=3,
                 radius=3):
        super().__init__()
        self.num_classes = num_classes
        self.nms = Radius_NMS(num_classes, min_height,
                              max_height, radius).cuda()
        self.k_means_iters = k_means_iters
        self.radius = radius

    def extract_feat(self, lidar_img, rgb_img=None, verbose=False):
        if verbose:
            start = perf_counter()
        lidar_semantics = self.lidar_cnn(lidar_img)
        if self.use_rgb:
            rgb_semantics = self.rgb_cnn.forward(rgb_img)
            fused_semantics = torch.cat((rgb_semantics, lidar_semantics),
                                        dim=1)
            out = self.deep_aggregation.forward(fused_semantics)
        else:
            out = self.deep_aggregation.forward(lidar_semantics)
        if verbose:
            print(f"extract_feat time: {(perf_counter() - start)/1e6} ms")
        return out

    def get_bboxes(self, feats, lidar_img, verbose=False):
        batch_size = lidar_img.shape[0]
        lidar_copy = lidar_img.clone()

        # Apply SoftNMS
        box_predictions = self.nms.forward(
            feats.detach().cpu(),
            lidar_copy.detach().cpu(),
            verbose=verbose
        )

        # Refine Predictions using K-means
        k_means_predictions = []
        start = perf_counter()
        _, ind = feats.max(dim=1)
        for b in range(batch_size):
            class_preds = []
            for c in range(1, self.num_classes):
                boxes_tensor = box_predictions[b][c-1]
                if (not isinstance(boxes_tensor, list)) \
                        and boxes_tensor.shape[0] > 0:
                    start_preds = boxes_tensor
                    num_of_cones = start_preds.shape[0]
                    class_cond = ind[b] == c
                    lidar_points_x = lidar_img[b][0][class_cond].reshape(-1, 1)
                    lidar_points_y = lidar_img[b][1][class_cond].reshape(-1, 1)

                    class_lidar_points = torch.cat(
                        (lidar_points_x, lidar_points_y), dim=1)
                    cl, k_preds = kmeans(
                        X=class_lidar_points, cluster_centers=start_preds,
                        num_clusters=num_of_cones,
                        iters=self.k_means_iters)
                    class_preds.append(k_preds)
                else:
                    class_preds.append([])
            k_means_predictions.append(class_preds)
        box_predictions = k_means_predictions
        k_means_time = perf_counter()-start
        if verbose:
            print(f"K-means refinement time: {k_means_time/1e6} ms")
        return box_predictions

    def forward_train(self, ):

    def simple_test(self, )
