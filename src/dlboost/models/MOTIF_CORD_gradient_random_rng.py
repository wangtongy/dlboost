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
        device0 = torch.device("cuda:2"),
    ):
        super().__init__()
        self.device0 = device0
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
        image_loss = self.gradient_image_loss(image_init, image_init_target) #0.8407
        image = image_init*csm
        hybrid_kspace = nufft_2d(image,kspace_traj,(320,320),norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)))
        kspace_estimated = fft_1D(hybrid_kspace,dim=2,norm="ortho")
        recon_loss = self.outer_loss(
            weights,kspace_estimated, kspace_data,True
        )  ## data consistency loss
        loss = torch.mean(recon_loss) # 0.0079 * = 0.79
        Final_loss = image_loss + 100*loss
        return Final_loss

    def gradient_image_loss(self,input_img, target_img):
        
        # Get gradients for input and target
        input_img_intensity = input_img.abs()
        target_img_intensity = target_img.abs()
        mag_grad_in = self.compute_gradient_3d(input_img_intensity)
        mag_grad_tg = self.compute_gradient_3d(target_img_intensity)

        intensity_in = input_img.abs()
        intensity_tg = target_img.abs()
        
        L_intensity = torch.mean(self.loss_fn(intensity_in, intensity_tg))
        # Example: L1 loss on gradients
        L_gradient = torch.mean(self.loss_fn(mag_grad_in, mag_grad_tg))
        
        L_img = 2*L_intensity + L_gradient
        
        return L_img
    
    def compute_gradient_3d(self,img):
        """
        Computes 3D gradients (grad_x, grad_y, grad_z) of 'img' using Sobel filters.
        img should have shape: (batch_size, channels, D, H, W).
        Returns:
            grad_x, grad_y, grad_z (each shape: (batch_size, channels, D, H, W)).
        """
        device = img.device
        # 1D filters for Sobel construction
        smoothing_1d = torch.tensor([1., 2., 1.])
        derivative_1d = torch.tensor([-1., 0., 1.])

        # Build Sobel kernel for derivative in x, smoothing in y & z
        sobel_x = torch.zeros((3, 3, 3),device=device)
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    sobel_x[z, y, x] = derivative_1d[x] * smoothing_1d[y] * smoothing_1d[z]
        sobel_x = sobel_x.view(1, 1, 3, 3, 3)

        # Build Sobel kernel for derivative in y, smoothing in x & z
        sobel_y = torch.zeros((3, 3, 3),device=device)
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    sobel_y[z, y, x] = smoothing_1d[x] * derivative_1d[y] * smoothing_1d[z]
        sobel_y = sobel_y.view(1, 1, 3, 3, 3)

        # Build Sobel kernel for derivative in z, smoothing in x & y
        sobel_z = torch.zeros((3, 3, 3),device=device)
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    sobel_z[z, y, x] = smoothing_1d[x] * smoothing_1d[y] * derivative_1d[z]
        sobel_z = sobel_z.view(1, 1, 3, 3, 3)
        # Repeat each kernel for all input channels, if necessary
        channels = img.shape[1]  # number of channels # b ch z h w
        sobel_x = sobel_x.repeat(channels, 1, 1, 1, 1)  # shape: (channels, 1, 3, 3, 3)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1, 1)
        sobel_z = sobel_z.repeat(channels, 1, 1, 1, 1)

        # 3D Convolution with padding=1 to keep same spatial/depth size
        grad_x = F.conv3d(img, sobel_x, padding=1, groups=channels)
        grad_y = F.conv3d(img, sobel_y, padding=1, groups=channels)
        grad_z = F.conv3d(img, sobel_z, padding=1, groups=channels)

        mag_grad = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return mag_grad

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
        nufft_im_size: tuple = (320, 320),
        epsilon: float = 1e-2,
        iterations: int = 5,
        gamma_init=0.01,
        tau_init=0.2,
    ):
        super().__init__()
        self.forward_model = MR_Forward_Model_Static(nufft_im_size)
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
            breakpoint()
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