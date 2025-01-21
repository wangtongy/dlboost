import einx
import torch
from torch import nn
import numpy as np
from dlboost.models import ComplexUnet, DWUNet, SpatialTransformNetwork
from dlboost.NODEO.Utils import resize_deformation_field
from dlboost.utils.tensor_utils import interpolate
from mrboost.computation import generate_nufft_op, nufft_2d, nufft_adj_2d
from pytorch_lightning import LightningModule
from mrboost.computation import (
    kspace_point_to_radial_spokes,
    radial_spokes_to_kspace_point,
)


class CSM_FixPh(nn.Module):
    def __init__(self):
        super().__init__()

    def generate_forward_operator(self, csm_kernels):
        self._csm = csm_kernels

    def forward(self, image):
        # breakpoint()
        return image * self._csm
# (1,1,8,320,320)*(1,20,8,320,320) = (1,20,8,320,320)

class NUFFT(nn.Module):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size
        self.norm_factor = 2 * np.sqrt(np.prod(self.nufft_im_size))

    def generate_forward_operator(self, kspace_traj):
        self.kspace_traj = kspace_traj

    def adjoint(self, kspace_data):
        return nufft_adj_2d(
            kspace_data,
            self.kspace_traj,
            self.nufft_im_size,
            norm_factor=self.norm_factor,
        )

    def forward(self, image):
        return nufft_2d(
            image, self.kspace_traj, self.nufft_im_size, norm_factor=self.norm_factor
        )


class MR_Forward_Model_Static(nn.Module):
    def __init__(
        self,
        image_size,
        nufft_im_size,
        CSM_module=CSM_FixPh,
        NUFFT_module=NUFFT,
    ):
        super().__init__()
        self.S = CSM_module()
        self.N = NUFFT_module(nufft_im_size)

    def generate_forward_operators(self, csm_kernels, kspace_traj):
        self.S.generate_forward_operator(csm_kernels)
        self.N.generate_forward_operator(kspace_traj)

    def forward(self, image):
        #_image = image.clone()
        image_multi_ch = self.S(image) #Coil sensitivity * image
        kspace_data_estimated = self.N(image_multi_ch) # S* F * C * Image
        return kspace_data_estimated


class Regularization(nn.Module):
    def __init__(self, pretrained_path="/bmrc-an-data/TongyaoW/Reconstruction/AcceleratedMR/Undersample/blackbone/experiments/BB_N2N_pretrain/N2N_Stationary_epoch=02.ckpt"):
        super().__init__()
        self.image_denoiser = ComplexUnet(
            1,
            1,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                features=(16, 32, 64, 128, 256),
                # features=(32, 64, 128, 256, 512),
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=(
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
            ),
            norm_with_given_std=True,
        )
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def forward(self, params, std):
        std = std.to(torch.float32)
        return self.image_denoiser(params, std=std)

    def load_pretrained(self, path):
        pretrained_state_dict = torch.load(path)
        self.image_denoiser.state_dict(pretrained_state_dict["state_dict"])


class Identity_Regularization:
    def __init__(self):
        self.recon_module = ComplexUnet(
            1,
            1,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                features=(16, 32, 64, 128, 256),
                # features=(32, 64, 128, 256, 512),
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=(
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
            ),
            norm_with_given_std=True,
        )
        self.path = "/bmrc-an-data/TongyaoW/Reconstruction/\
AcceleratedMR/Undersample/blackbone/experiments/BB_N2N_pretrain/N2N_Stationary_epoch=02.ckpt"
        self.recon_module.state_dict(
            torch.load(self.path, map_location="cuda")["state_dict"]
        )
        self.recon_module = self.recon_module.cuda()

    def __call__(self, params, std=None):
        return self.recon_module(params, std=std)


class MOTIF_CORD(nn.Module):
    def __init__(
        self,
        patch_size: tuple = (16, 320, 320),
        nufft_im_size: tuple = (320, 320),
        epsilon: float = 1e-2,
        iterations: int = 5,
        gamma_init=0.01,
        tau_init=0.2,
    ):
        super().__init__()
        self.forward_model = MR_Forward_Model_Static(patch_size, nufft_im_size)
        self.regularization = Regularization()
        self.epsilon = epsilon
        self.iterations = iterations
        #self.gamma = nn.Parameter(gamma_init*torch.ones(iterations))
        self.gamma = gamma_init
        self.tau = nn.Parameter(tau_init * torch.ones(iterations))
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        # self.loss_fn = nn.MSELoss(reduction="mean") ## L2 loss
        self.loss_fn = torch.nn.L1Loss(reduction="mean")
        # self.nufft_adj = tkbn.KbNufftAdjoint(im_size=nufft_im_size)

    def forward(
        self,
        kspace_data,
        kspace_traj,
        image_init,
        csm,
        std,
        weights_flag=True,
    ):
        image_init = torch.nan_to_num_(image_init) 
        image_list = [] 
        x = image_init
        image_list.append(image_init.cpu())
        self.forward_model.generate_forward_operators(csm, kspace_traj) #output kspace estimated

        x.requires_grad_(True) 
        #estimated kspace will be generated inthe inner_loss function
        for t in range(self.iterations):
            print("iteration", t, "start")
            dc_loss = self.inner_loss(
                x.clone(), kspace_data, kspace_traj,weights_flag
            )  ## data consistency loss
            grad_dc = torch.autograd.grad(dc_loss, x)[0]
            grad_reg = x - self.regularization(x, std=std)
            updates = -self.gamma * (grad_dc + self.tau[t] * grad_reg)
            #updates = -(self.gamma * grad_dc)
            mean_grad_dc_real = torch.mean(grad_dc.real)
            mean_grad_reg = torch.mean(grad_reg.real)
            mean_grad_dc_imag = torch.mean(grad_dc.imag)
            mean_grad_reg_imag = torch.mean(grad_reg.imag)
            # ic(self.gamma)
            ic(self.tau[t]) 
            x = x.add(updates) #batch, channel, z, h,w
            image_list.append(x.clone().detach().cpu()) #itr, b,c,z,h,w
            print(f"t: {t}, innerloss: {dc_loss}")
            print(f"t:{t}, gdc_real = {mean_grad_dc_real}, gdc_imag = {mean_grad_dc_imag},greg_real = {mean_grad_reg},greg_imag = {mean_grad_reg_imag}")
        return x, image_list
 
    def inner_loss(self, x, kspace_data,kspace_traj,weights_flag): 
        kspace_data_estimated = self.forward_model(x) #x^
        if weights_flag:
            # kspace_data_estimated_detatched = (
            #     kspace_data_estimated.detach().abs()
            # )
            # norm_factor = kspace_data_estimated_detatched.max()
            # weights = 1 / (kspace_data_estimated_detatched / norm_factor + self.epsilon)
            kspace_traj = kspace_point_to_radial_spokes(kspace_traj,640)
            kspace_traj_norm = kspace_traj/kspace_traj.abs().max()/2
            kx,ky = kspace_traj_norm[0]
            kx_center,ky_center = kx.mean(),ky.mean()
            distances = torch.sqrt((kx - kx_center) ** 2 + (ky - ky_center) ** 2) 
            weights = distances/distances.max()+1 ## 1 is the epison to avoid zero division
            weights = radial_spokes_to_kspace_point(weights)
        else:
            weights = 1
 
        loss_dc = self.loss_fn(
            torch.view_as_real(weights * kspace_data_estimated),
            torch.view_as_real(weights * kspace_data),
        )
      
        #self.log("DC_loss", loss_dc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_dc
