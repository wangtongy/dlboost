import einx
import torch
from torch import nn
import numpy as np
from dlboost.models import ComplexUnet, DWUNet, SpatialTransformNetwork
from dlboost.NODEO.Utils import resize_deformation_field
from dlboost.utils.tensor_utils import interpolate
from mrboost.computation import nufft_2d, nufft_adj_2d, fft_1D
from pytorch_lightning import LightningModule
import torch.nn.functional as F

class CSM_FixPh(nn.Module):
    def __init__(self):
        super().__init__()

    def generate_forward_operator(self, csm_kernels):
        self._csm = csm_kernels

    def forward(self, image):
        return image * self._csm 
# csm: (1,ch,z,h,w)

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

class MOTIF_Pretrain(nn.Module):
    def __init__(
        self,
        nufft_im_size: tuple = (320, 320),
    ):
        super().__init__()
        self.device0 = torch.device("cuda:1")
        self.nufft_im_size = nufft_im_size
        # self.forward_model = MR_Forward_Model(nufft_im_size).to(self.device0)
        self.loss_fn = nn.L1Loss(reduction="mean") ## L1 loss
        # self.loss_fn = nn.MSELoss(reduction = "mean") # L2 loss

    def forward(
        self,
        kspace_data,
        weights,
        kspace_traj,
        image_init, # z ch h w
        image_init_target,
        csm,
        weights_flag=True,
    ):

        image_init = torch.nan_to_num_(image_init).to(self.device0)
        image_init_target = torch.nan_to_num_(image_init_target).to(self.device0)
        csm = csm.to(self.device0)
        kspace_traj = kspace_traj.to(self.device0)
        kspace_data = kspace_data.to(self.device0)
        weights = weights.to(self.device0)
        ### Image Loss
        ssim_loss = 1-self.ssim_3d(image_init, image_init_target) #0.8407
    
        image = image_init*csm
        hybrid_kspace = nufft_2d(image,kspace_traj,(320,320),norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)))
        kspace_estimated = fft_1D(hybrid_kspace,dim=2,norm="ortho")
        recon_loss = self.outer_loss(
            weights,kspace_estimated, kspace_data,True
        )  ## data consistency loss
        loss = torch.mean(recon_loss) # 0.0079 * = 0.79
        Final_loss = ssim_loss + 100*loss
        return Final_loss

    def create_3d_gaussian_kernel(self,win_size, sigma, channels, device, dtype=torch.float32):
        """
        Creates a 5D Gaussian filter for 3D convolution:
        shape = (channels, 1, win_size, win_size, win_size).
        """
        # 1D coordinates centered at 0
        coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1)/2.0
        # 1D Gaussian
        g_1d = torch.exp(-coords**2/(2*sigma**2))
        g_1d /= g_1d.sum()
        
        # Outer products to form 3D kernel
        g_3d = g_1d[:, None, None] * g_1d[None, :, None] * g_1d[None, None, :]
        g_3d /= g_3d.sum()  # normalize

        # Reshape for conv3d => [out_channels=channels, in_channels=1, D, H, W]
        g_3d = g_3d.view(1, 1, win_size, win_size, win_size)
        g_3d = g_3d.repeat(channels, 1, 1, 1, 1)
        return g_3d

    def ssim_3d(
            self,
        vol1: torch.Tensor,
        vol2: torch.Tensor,
        win_size: int = 3,
        sigma: float = 1.0,
        data_range: float = None
    ) -> torch.Tensor:
        """
        Computes 3D SSIM between two volumes of shape [H, W, D].
        Returns a scalar in [0,1], where 1 = identical.
        
        Args:
            vol1, vol2: shape [H, W, D]. (Single-channel 3D volumes)
            win_size: size of the 3D Gaussian kernel. Typically 3 or 5 for 3D.
            sigma: std for Gaussian kernel.
            data_range: difference between max and min of volumes.
                        If None, it is computed from vol1 & vol2.
        """
        # 1) Unsqueeze to [B=1, C=1, H, W, D]
        vol1 = vol1.abs()
        vol2 = vol2.abs()

        dtype = vol1.dtype

        # 2) Determine data_range if not provided
        if data_range is None:
            min_val = torch.min(vol1.min(), vol2.min())
            max_val = torch.max(vol1.max(), vol2.max())
            data_range = (max_val - min_val).clamp_min(1e-8)

        # SSIM constants (following the original paper)
        K1, K2 = 0.01, 0.03
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        # 3) Build 3D Gaussian kernel
        channels = 1  # single-channel
        window_3d = self.create_3d_gaussian_kernel(
            win_size, sigma, channels, self.device0, dtype
        )

        # 4) Compute local means via 3D convolution
        mu1 = F.conv3d(vol1, window_3d, padding=win_size//2, groups=channels)
        mu2 = F.conv3d(vol2, window_3d, padding=win_size//2, groups=channels)

        mu1_sq   = mu1.pow(2)
        mu2_sq   = mu2.pow(2)
        mu1_mu2  = mu1 * mu2

        # 5) Compute local variances & covariance
        sigma1_sq = F.conv3d(vol1 * vol1, window_3d, padding=win_size//2, groups=channels) - mu1_sq
        sigma2_sq = F.conv3d(vol2 * vol2, window_3d, padding=win_size//2, groups=channels) - mu2_sq
        sigma12   = F.conv3d(vol1 * vol2, window_3d, padding=win_size//2, groups=channels) - mu1_mu2

        # 6) SSIM formula
        num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (den + 1e-12)

    # 7) Return mean SSIM over the 3D volume
        return ssim_map.mean()

    def outer_loss(self, weights,kspace_data_estimated, kspace_data,weights_flag): 
        
        ic(kspace_data_estimated.mean())
        ic(kspace_data.mean())
        if weights_flag:
            weights = weights
        else:
            weights = 1
        ic(kspace_data_estimated.shape)
        if kspace_data_estimated.shape[1] == 64:
            kspace_data_estimated = torch.split(kspace_data_estimated, 32, dim=1)
            kspace_data = torch.split(kspace_data, 32, dim=1)
            loss_total = []
            for i in range(2):
                loss_dc = self.loss_fn(
                    torch.view_as_real(weights * kspace_data_estimated[i]),# [b, ch, z, length] * [b, z, length]
                    torch.view_as_real(weights * kspace_data[i]),
                )
                loss_total.append(loss_dc)
            loss = torch.stack(loss_total).mean()

        else:
            loss = self.loss_fn(
                torch.view_as_real(weights * kspace_data_estimated),# [b, ch, z, length] * [b, z, length]
                torch.view_as_real(weights * kspace_data),
            )
        return loss


class MR_Forward_Model(nn.Module):
    def __init__(
        self,
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
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.image_denoiser = ComplexUnet(
            1,
            1,
            spatial_dims=3,
            # conv_net=None,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                # features=(8, 16, 32, 64, 128),
                features=(32, 64, 128, 256, 512),
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
            conv_net=None,
            norm_with_given_std=True,
        )
        self.path = "/bmrc-an-data/TongyaoW/Reconstruction/\
AcceleratedMR/Undersample/MOTIF_CORD_ty/experiments/MOTIF_CORD_random_SE_Pretrain_Unet/MOTIF_CORD_epoch=49.ckpt"
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
        # self.forward_model = MR_Forward_Model(nufft_im_size)
        self.regularization = Identity_Regularization()
        self.epsilon = epsilon
        self.iterations = iterations
        #self.gamma = nn.Parameter(gamma_init*torch.ones(iterations))
        self.gamma = gamma_init
        self.tau = nn.Parameter(tau_init * torch.ones(iterations))
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.loss_fn = nn.MSELoss(reduction="mean") ## L2 loss
        # self.nufft_adj = tkbn.KbNufftAdjoint(im_size=nufft_im_size)


    def forward(
        self,
        weights,
        kspace_data,
        kspace_traj,
        image_init, # z ch h w
        csm,
        std,
        weights_flag=True,
    ):
        image_init = torch.nan_to_num_(image_init) 
        image_list = [] 
        x = image_init

        image_list.append(image_init.cpu())
        ic(csm.shape) # 1 z ch h w
        ic(kspace_traj.shape) # 1 2 length
        self.forward_model.generate_forward_operators(csm, kspace_traj) #output kspace estimated
        ic(x.shape) # 1 z ch, h w
        x.requires_grad_(True) 
        #estimated kspace will be generated inthe inner_loss function
        for t in range(self.iterations):
            print("iteration", t, "start")
            
            dc_loss = self.inner_loss(
                weights,x.clone(), kspace_data,True
            )  ## data consistency loss
   
            grad_dc = torch.autograd.grad(dc_loss, x)[0]
            # x_s = x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4]) # b,ch,z,h,w -> b,z,ch,h*w
            ic(grad_dc.shape)
            ic(x.shape) # b,z,ch,h,w # 1,5,1,320,320
            grad_reg = x - self.regularization(x, std=std)
            # grad_reg = grad_reg.view(grad_reg.shape[0], grad_reg.shape[2], grad_reg.shape[1], grad_reg.shape[3], grad_reg.shape[4])
            ic(grad_reg.shape)
            updates = -self.gamma * (grad_dc + self.tau[t] * grad_reg)
            mean_grad_dc_real = torch.mean(grad_dc.real)
            mean_grad_reg = torch.mean(grad_reg.real)
            mean_grad_dc_imag = torch.mean(grad_dc.imag)
            mean_grad_reg_imag = torch.mean(grad_reg.imag)
            # ic(self.gamma)
            ic(self.tau[t]) 
            # x = x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4]) # b,z,ch,h,w
            x = x.add(updates) # 1,5,1,320,320
            ############### only turn it on if needed during testing ##############
            image_list.append(x.clone().detach().cpu()) #itr, b,c,z,h,w
            print(f"t: {t}, innerloss: {dc_loss}")
            print(f"t:{t}, gdc_real = {mean_grad_dc_real}, gdc_imag = {mean_grad_dc_imag},greg_real = {mean_grad_reg},greg_imag = {mean_grad_reg_imag}")
        return x, image_list
        #return x
 
    # def inner_loss(self, weights,x, kspace_data,weights_flag): 
    #     kspace_data_estimated = self.forward_model(x) #x^
    #     kspace_estimated = fft_1D(kspace_data_estimated,dim=2,norm="ortho")
    #     if weights_flag:
    #         weights = weights
    #     else:
    #         weights = 1
    #     breakpoint()
    #     # kspace_data_estimated = einx.rearrange("b z ch length -> b ch z length", kspace_data_estimated)
    #     # kspace_data = einx.rearrange("b z ch length -> b ch z length", kspace_data)
    #     loss_dc = self.loss_fn(
    #         torch.view_as_real(weights * kspace_estimated),# [b, ch, z, length] * [b, z, length]
    #         torch.view_as_real(weights * kspace_data),
    #     )
      
    #     #self.log("DC_loss", loss_dc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss_dc

    def inner_loss(self, weights,x, kspace_data,weights_flag): 
            kspace_data_estimated = self.forward_model(x) #x^
            breakpoint()
            if weights_flag:
                weights = weights
            else:
                weights = 1
            loss_dc = self.loss_fn(
                torch.view_as_real(weights * kspace_data_estimated),# [b, ch, z, length] * [b, z, ch, length]
                torch.view_as_real(weights * kspace_data),
            )
        
            #self.log("DC_loss", loss_dc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss_dc