"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import torchvision.transforms as transforms
import torch
import pdb

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size
        
        
        transform_list = [transforms.Resize((35,35), Image.NEAREST),
        transforms.ToTensor(),
                       ]
        self.trans35 = transforms.Compose(transform_list)

        transform_list = [transforms.Resize((19,19), Image.NEAREST),
            transforms.ToTensor(),
                           ]
        self.trans19 = transforms.Compose(transform_list)

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        smap35 = self.trans35(label) * 255
        smap19 = self.trans19(label) * 255
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        label_map_segmentation = label_tensor.clone().type(torch.long)


        _, h, w = label_tensor.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = torch.FloatTensor(nc, h, w).zero_()
        label_tensor = input_label.scatter_(0, label_tensor.long(), 1.0)
        if self.opt.c2f or self.opt.c2f_sem or self.opt.c2f_sem_rec:
            # create one-hot label for loss
            onnnnes = torch.FloatTensor(1, h, w).fill_(1)
            coarse_fine = torch.unsqueeze(torch.cat((label_tensor, onnnnes), dim = 0), 0)
            if self.opt.aspect_ratio==2.0 and self.opt.load_size==512:
                smap35 = torch.squeeze(torch.nn.functional.interpolate(coarse_fine, size=(35,67), mode='nearest'), dim=0)
                smap19 = torch.squeeze(torch.nn.functional.interpolate(coarse_fine, size=(19,35), mode='nearest'), dim=0)
            elif self.opt.aspect_ratio==1.0: #the conditions should be completed
                smap35 = torch.squeeze(torch.nn.functional.interpolate(coarse_fine, size=(35,35), mode='nearest'), dim=0)
                smap19 = torch.squeeze(torch.nn.functional.interpolate(coarse_fine, size=(19,19), mode='nearest'), dim=0)
            else:
                raise 'Not implemented semantic shapes'

            if self.opt.normalize_smaps:
                r = torch.mean(smap35[:-1,:,:]+1e-9, [1,2], keepdim=True)
                smap35[:-1,:,:] = self.opt.fine_grained_scale*(1/nc)*smap35[:-1,:,:]/r
                r = torch.mean(smap19[:-1,:,:]+1e-9, [1,2], keepdim=True)
                smap19[:-1,:,:] = self.opt.fine_grained_scale*(1/nc)*smap19[:-1,:,:]/r


        

        tmp = ((label_map_segmentation == 0) + (label_map_segmentation == 1) + (label_map_segmentation == 2) + (label_map_segmentation == 3) + (label_map_segmentation == 4) + (label_map_segmentation == 5) + (label_map_segmentation == 6) + (label_map_segmentation == 9) + (label_map_segmentation == 10) \
                    + (label_map_segmentation == 14) + (label_map_segmentation == 15) + (label_map_segmentation == 16) + (label_map_segmentation == 18) + (label_map_segmentation == 29) + (label_map_segmentation == 30) + (label_map_segmentation == -1)).type(torch.bool)
        label_map_segmentation[tmp] = 255

        label_map_segmentation[label_map_segmentation==7]=0
        label_map_segmentation[label_map_segmentation==8]=1
        label_map_segmentation[label_map_segmentation==11]=2
        label_map_segmentation[label_map_segmentation==12]=3
        label_map_segmentation[label_map_segmentation==13]=4
        label_map_segmentation[label_map_segmentation==17]=5
        label_map_segmentation[label_map_segmentation==19]=6
        label_map_segmentation[label_map_segmentation==20]=7
        label_map_segmentation[label_map_segmentation==21]=8
        label_map_segmentation[label_map_segmentation==22]=9
        label_map_segmentation[label_map_segmentation==23]=10
        label_map_segmentation[label_map_segmentation==24]=11
        label_map_segmentation[label_map_segmentation==25]=12
        label_map_segmentation[label_map_segmentation==26]=13
        label_map_segmentation[label_map_segmentation==27]=14
        label_map_segmentation[label_map_segmentation==28]=15
        label_map_segmentation[label_map_segmentation==31]=16
        label_map_segmentation[label_map_segmentation==32]=17
        label_map_segmentation[label_map_segmentation==33]=18

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'smap35': smap35,
                      'smap19':smap19,
                      'smapS':label_map_segmentation
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
