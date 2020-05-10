# def gauss_kernel(kernlen=21, nsig=3, channels=1):
#     import numpy as np
#     import scipy.stats as st
#     interval = (2*nsig+1.)/(kernlen)
#     x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw/kernel_raw.sum()
#     out_filter = np.array(kernel, dtype = np.float32)
#     out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
#     out_filter = np.repeat(out_filter, channels, axis = 2)
#     return out_filter
#
# # def blur(x):
# #     import tensorflow as tf
# #     kernel_var = gauss_kernel(21, 3, 3)
# #     return tf.nn.depthwise_conv2d(x.detach(), kernel_var, [1, 1, 1, 1], padding='SAME')
#
# def cnn2d_depthwise_torch(image: np.ndarray,
#                           filters: np.ndarray):
#     from torch.nn import functional as F
#     image_torch, filters_torch = convert_to_torch(image, filters)
#     df, _, cin, cmul = filters.shape
#     filters_torch = filters_torch.transpose(0, 1).contiguous()
#     filters_torch = filters_torch.view(cin * cmul, 1, df, df)
#
#     features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2, groups=cin)
#     features_torch_ = features_torch.numpy()[0].transpose([2, 1, 0])
#
#     return features_torch_
import torch.nn as nn
import torch
from torchvision import models
import math
import numbers
from torch import nn
from torch.nn import functional as F

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

class VGGLoss(object):
    def __init__(self, device):
        self.vgg = VGGNet().to(device).eval()

    def loss(self, out_img, gt_img):
        content_out = self.vgg(out_img)
        content_gt = self.vgg(gt_img)
        content_loss = 0
        for f1, f2 in zip(content_out, content_gt):
            # Compute content loss with output and ground truth images
            content_loss += torch.mean((f1 - f2)**2)
        return content_loss

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Example:
    smoothing = GaussianSmoothing(3, 5, 1)
    input = torch.rand(1, 3, 100, 100)
    input = F.pad(input, (2, 2, 2, 2), mode='reflect')
    output = smoothing(input)
    """
    def __init__(self, channels, kernel_size, sigma, device, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.to(device))
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

