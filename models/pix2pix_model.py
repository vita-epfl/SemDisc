"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
import pdb
import numpy as np

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt, wmse = (self.opt.c2f_sem_rec or self.opt.c2f or self.opt.c2f_sem))
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255)
            self.upsample = torch.nn.Upsample(size=(int(self.opt.crop_size//self.opt.aspect_ratio), self.opt.crop_size), mode='bilinear')
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.use_weight_decay:
                self.WDLoss = torch.nn.MSELoss()


    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        #pdb.set_trace()
        if self.opt.c2f:
            self.smap = []
            input_semantics, real_image, smap35, smap19 = self.preprocess_input(data)
            self.smap.append(smap35)
            self.smap.append(smap19)
            label_mapS = None
        elif self.opt.c2f_sem or self.opt.c2f_sem_rec:
            self.smap = []
            input_semantics, real_image, smap35, smap19, label_mapS = self.preprocess_input(data)
            self.smap.append(smap35)
            self.smap.append(smap19)
        else:
            input_semantics, real_image = self.preprocess_input(data)
            self.smap = None
            label_mapS = None
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, fine_tune = self.opt.fine_tune, label_mapS = label_mapS)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, label_mapS = label_mapS)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                if self.opt.netG=='asapnets':
                    fake_image, _, _ = self.generate_fake(input_semantics, real_image)
                else:
                    fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if (opt.isTrain or opt.D_output) else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if (opt.isTrain or opt.D_output):
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        #data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            data['smap35'] = data['smap35'].cuda()
            data['smap19'] = data['smap19'].cuda()
            data['smapS'] = data['smapS'].cuda()

        if self.opt.D_output:
            self.image_paths = data['path']


        # create one-hot label map
        label_map = data['label']
        label_mapS = torch.squeeze(data['smapS'], dim=1)

        input_semantics = label_map
        smap35 = data['smap35']
        smap19 = data['smap19']

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
            
        if self.opt.c2f:
            return input_semantics, data['image'], smap35, smap19
        elif self.opt.c2f_sem or self.opt.c2f_sem_rec:
            return input_semantics, data['image'], smap35, smap19, label_mapS
        else:
            return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image, smap35 = None, smap19 = None, fine_tune = False, label_mapS = None):
        
        G_losses = {}

        if self.opt.netG=='asapnets':
            fake_image, lr_features, KLD_loss = self.generate_fake(
                input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
            if self.opt.use_weight_decay:
                lr_features_l2 = lr_features.norm(p=2)
                device = lr_features_l2.device
                zero = torch.zeros(lr_features_l2.shape).to(device)
                G_losses['WD'] = self.WDLoss(lr_features_l2, zero) \
                                 * self.opt.lambda_WD
        else:
            fake_image, KLD_loss = self.generate_fake(
                input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        #pdb.set_trace()
        if self.opt.c2f_sem:
            pred_fake, pred_real, pred_fakeS, pred_realS = self.discriminate(
                input_semantics, fake_image, real_image)
        elif self.opt.c2f_sem_rec:
            pred_fake, pred_real, pred_fakeS, pred_realS, pred_fakeSelf, pred_realSelf = self.discriminate(
                input_semantics, fake_image, real_image)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,self.smap,
                                            for_discriminator=False) * self.opt.lambda_GAN

        # Segmentation part (including self-supervised and semantic-segmentation losses)
        if self.opt.c2f_sem:
            if type(pred_fakeS) == list:
                G_losses['Seg'] = 0.0
                for seg in pred_fakeS: # different discriminators for different scales
                    seg = self.upsample(seg)
                    G_losses['Seg'] = G_losses['Seg'] + self.criterionSeg(seg, label_mapS) * self.opt.lambda_seg * (0.0+self.opt.active_GSeg)
            else:
                pred_fakeS = self.upsample(pred_fakeS)
                G_losses['Seg'] = self.criterionSeg(pred_fakeS, label_mapS) * self.opt.lambda_seg * (self.opt.active_GSeg)
        elif self.opt.c2f_sem_rec:
            if type(pred_fakeS) == list:
                G_losses['Seg'] = 0.0
                for seg in pred_fakeSelf: # different discriminators for different scales
                    seg = self.upsample(seg)
                    G_losses['Seg'] = G_losses['Seg'] + self.criterionFeat(seg, real_image) * self.opt.lambda_rec * (0.0+self.opt.active_GSeg)
                for seg in pred_fakeS: # different discriminators for different scales
                    seg = self.upsample(seg)
                    G_losses['Seg'] = G_losses['Seg'] + self.criterionSeg(seg, label_mapS) * self.opt.lambda_seg * (0.0+self.opt.active_GSeg)
            else:
                pred_fakeS = self.upsample(pred_fakeS)
                G_losses['Seg'] = self.criterionSeg(pred_fakeS, label_mapS) * self.opt.lambda_seg * (self.opt.active_GSeg)
        elif fine_tune:
            self.real_cf = self.netS(self.m(real_image))
            self.fake_cf = self.netS(self.m(fake_image))
            G_losses['Seg'] = self.criterionFeat(self.real_cf,self.fake_cf) * self.opt.lambda_seg
            
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, label_mapS = None):
        D_losses = {}
        with torch.no_grad():
            if self.opt.netG=='asapnets':
                fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            else:
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if self.opt.c2f_sem_rec:
            pred_fake, pred_real, pred_fakeS, pred_realS, pred_fakeSelf, pred_realSelf = self.discriminate(
                input_semantics, fake_image, real_image)
            if type(pred_realS) == list:
                D_losses['D_Seg'] = 0
                for seg in pred_realSelf: # different discriminators for different scales
                    seg = self.upsample(seg)
                    D_losses['D_Seg'] = D_losses['D_Seg'] + self.criterionFeat(seg, real_image) * self.opt.lambda_rec
                for seg in pred_realS: # different discriminators for different scales
                    seg = self.upsample(seg)
                    D_losses['D_Seg'] = D_losses['D_Seg'] + self.criterionSeg(seg, label_mapS) * self.opt.lambda_seg
            else:
                pred_realS = self.upsample(pred_realS)
                D_losses['D_Seg'] = self.criterionFeat(pred_realS, real_image) * self.opt.lambda_seg
        elif self.opt.c2f_sem:
            pred_fake, pred_real, pred_fakeS, pred_realS = self.discriminate(
                input_semantics, fake_image, real_image)
            if type(pred_realS) == list:
                D_losses['D_Seg'] = 0
                for seg in pred_realS: # different discriminators for different scales
                    seg = self.upsample(seg)
                    D_losses['D_Seg'] = D_losses['D_Seg'] + self.criterionSeg(seg, label_mapS) * self.opt.lambda_seg
            else:
                pred_realS = self.upsample(pred_realS)
                D_losses['D_Seg'] = self.criterionFeat(pred_realS, real_image) * self.opt.lambda_seg
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        
        D_losses['D_fake'] = self.criterionGAN(pred_fake, False,self.smap,
                                               for_discriminator=True) * self.opt.lambda_GAN
        D_losses['D_real'] = self.criterionGAN(pred_real, True, self.smap,
                                               for_discriminator=True) * self.opt.lambda_GAN

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld


        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        if self.opt.netG=='asapnets':
            fake_image, lr_features = self.netG(input_semantics, z=z)
            return fake_image, lr_features, KLD_loss
        else:
            fake_image = self.netG(input_semantics, z=z)
            if self.opt.D_output:
                pred_fake, pred_real, pred_fakeS, pred_realS = self.discriminate(input_semantics, fake_image.detach(), real_image)
                np.save('vis/real0_'+self.image_paths[0].split('/')[-1], pred_real[0][-1].cpu().numpy())
                np.save('vis/fake0_'+self.image_paths[0].split('/')[-1], pred_fake[0][-1].cpu().numpy())
                np.save('vis/real1_'+self.image_paths[0].split('/')[-1], pred_real[1][-1].cpu().numpy())
                np.save('vis/fake1_'+self.image_paths[0].split('/')[-1], pred_fake[1][-1].cpu().numpy())
                np.save('vis/realS0_'+self.image_paths[0].split('/')[-1], pred_realS[0].cpu().numpy())
                np.save('vis/fakeS0_'+self.image_paths[0].split('/')[-1], pred_fakeS[0].cpu().numpy())
                np.save('vis/realS1_'+self.image_paths[0].split('/')[-1], pred_realS[1].cpu().numpy())
                np.save('vis/fakeS1_'+self.image_paths[0].split('/')[-1], pred_fakeS[1].cpu().numpy())
                np.save('vis/smap_'+self.image_paths[0].split('/')[-1], input_semantics.cpu().numpy())
            return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        if self.opt.c2f_sem or self.opt.c2f_sem_rec:
            fake_concat = fake_image
            real_concat = real_image
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        if self.opt.c2f_sem:
            pred_fake, pred_real, pred_fakeS, pred_realS = self.divide_pred_seg(discriminator_out)
            return pred_fake, pred_real, pred_fakeS, pred_realS
        elif self.opt.c2f_sem_rec:
            pred_fake, pred_real, pred_fakeS, pred_realS, pred_fakeSelf, pred_realSelf = self.divide_pred_seg_rec(discriminator_out)
            return pred_fake, pred_real, pred_fakeS, pred_realS, pred_fakeSelf, pred_realSelf
        else:
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            return pred_fake, pred_real

    # Considering both segmentation and discriminator concatenated
    def divide_pred_seg_rec(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            fakeS = []
            realS = []
            fakeSelf = []
            realSelf = []
            for p in pred:
                fake_tmp=[tensor[:tensor.size(0) // 2] for tensor in p[:-1]]
                real_tmp=[tensor[tensor.size(0) // 2:] for tensor in p[:-1]]
                tensor=p[-1]
                fake_tmp.append(tensor[:tensor.size(0) // 2, :-3-19])
                real_tmp.append(tensor[tensor.size(0) // 2:, :-3-19])
                fake.append(fake_tmp)
                real.append(real_tmp)
                fakeS.append(tensor[:tensor.size(0) // 2, -3-19:-3])
                realS.append(tensor[tensor.size(0) // 2:, -3-19:-3])
                fakeSelf.append(tensor[:tensor.size(0) // 2, -3:])
                realSelf.append(tensor[tensor.size(0) // 2:, -3:])
        else:
            print("not implemented yet")
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
            fakeS = pred[-3 + pred.size(0) // 2 : pred.size(0) // 2]
            realS = pred[-3:]

        return fake, real, fakeS, realS, fakeSelf, realSelf

    # Considering both segmentation and discriminator concatenated
    def divide_pred_seg(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            fakeS = []
            realS = []
            for p in pred:
                fake_tmp=[tensor[:tensor.size(0) // 2] for tensor in p[:-1]]
                real_tmp=[tensor[tensor.size(0) // 2:] for tensor in p[:-1]]
                tensor=p[-1]
                fake_tmp.append(tensor[:tensor.size(0) // 2, :-19])
                real_tmp.append(tensor[tensor.size(0) // 2:, :-19])
                fake.append(fake_tmp)
                real.append(real_tmp)
                fakeS.append(tensor[:tensor.size(0) // 2, -19:])
                realS.append(tensor[tensor.size(0) // 2:, -19:])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
            fakeS = pred[-19 + pred.size(0) // 2 : pred.size(0) // 2]
            realS = pred[-19:]

        return fake, real, fakeS, realS

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
