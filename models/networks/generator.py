"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

from models.networks.architecture import ASAPNetsBlock as ASAPNetsBlock
from models.networks.architecture import MySeparableBilinearDownsample as BilinearDownsample
from math import pi
from math import log2

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)



class ASAPNetsGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, hr_stream=None, lr_stream=None, fast=False):
        super(ASAPNetsGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)


        self.highres_stream = ASAPNetsHRStream(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        num_inputs_lr = self.highres_stream.num_inputs + (1 if opt.lr_instance else 0)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.lowres_stream = ASAPNetsLRStream(num_inputs_lr, num_params, norm_layer, width=opt.lr_width,
                                              max_width=opt.lr_max_width, depth=opt.lr_depth,
                                              learned_ds_factor=opt.learned_ds_factor,
                                              reflection_pad=opt.reflection_pad, **lr_stream)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im

    def forward(self, highres, z=None):
        lowres = self.get_lowres(highres)
        lr_features = self.lowres_stream(lowres)
        output = self.highres_stream(highres, lr_features)
        return output, lr_features#, lowres


def _get_coords(bs, h, w, device, ds, coords_type):
    """Creates the position encoding for the pixel-wise MLPs"""
    if coords_type == 'cosine':
        f0 = ds
        f = f0
        while f > 1:
            x = torch.arange(0, w).float()
            y = torch.arange(0, h).float()
            xcos = torch.cos((2 * pi * torch.remainder(x, f).float() / f).float())
            xsin = torch.sin((2 * pi * torch.remainder(x, f).float() / f).float())
            ycos = torch.cos((2 * pi * torch.remainder(y, f).float() / f).float())
            ysin = torch.sin((2 * pi * torch.remainder(y, f).float() / f).float())
            xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            coords_cur = torch.cat([xcos, xsin, ycos, ysin], 1).to(device)
            if f < f0:
                coords = torch.cat([coords, coords_cur], 1).to(device)
            else:
                coords = coords_cur
            f = f//2
    else:
        raise NotImplementedError()
    return coords.to(device)


class ASAPNetsLRStream(torch.nn.Sequential):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_in, num_out, norm_layer, width=64, max_width=1024, depth=7, learned_ds_factor=16,
                 reflection_pad=False, replicate_pad=False):
        super(ASAPNetsLRStream, self).__init__()

        model = []

        self.num_out = num_out
        padw = 1
        if reflection_pad:
            padw = 0
            model += [torch.nn.ReflectionPad2d(1)]
        if replicate_pad:
            padw = 0
            model += [torch.nn.ReplicationPad2d(1)]

        count_ly = 0

        model += [norm_layer(torch.nn.Conv2d(num_in, width, 3, stride=1, padding=padw)),
                  torch.nn.ReLU(inplace=True)]

        num_ds_layers = int(log2(learned_ds_factor))

        # strided conv layers for learning downsampled representation of the input"
        for i in range(num_ds_layers):
            if reflection_pad:
                model += [torch.nn.ReflectionPad2d(1)]
            if replicate_pad:
                model += [torch.nn.ReplicationPad2d(1)]
            if i == num_ds_layers-1:
                last_width = max_width
                model += [norm_layer(torch.nn.Conv2d(width, last_width, 3, stride=2, padding=padw)),
                          torch.nn.ReLU(inplace=True)]
                width = last_width
            else:
                model += [norm_layer(torch.nn.Conv2d(width, width, 3, stride=2, padding=padw)),
                      torch.nn.ReLU(inplace=True)]

        # ConvNet to estimate the MLPs parameters"
        for i in range(count_ly, count_ly+depth):
            model += [ASAPNetsBlock(width, norm_layer, reflection_pad=reflection_pad, replicate_pad=replicate_pad)]

        # Final parameter prediction layer, transfer conv channels into the per-pixel number of MLP parameters
        model += [torch.nn.Conv2d(width, self.num_out, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ASAPNetsHRStream(torch.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(ASAPNetsHRStream, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += int(4*log2(self.downsampling))
        self.channels = [in_ch]
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams += nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx += nco

            nparams += nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx += nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        # Fetch sizes
        k = int(self.downsampling)
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape

        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords(bs, h, w, highres.device, self.ds, self.coordinates)
            highres = torch.cat([highres, self.xy_coords], 1)


        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = torch.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)

        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out