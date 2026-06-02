from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from local_diffusion.data import DatasetBundle
from local_diffusion.models import register_model
from local_diffusion.models.optimal import OptimalDenoiser
from local_diffusion.models.base import BaseDenoiser



class LocalScoreMachineBase(BaseDenoiser):
    """Base class for all Local Score Machines from https://arxiv.org/pdf/2412.20292"""

    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        patch_lengths: list[int] = None,
        **kwargs: Any,
    ) -> None:
        params = params or {}
        super().__init__(
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )

        self.default_max_samps = params.get("default_max_samps", None)
        self.dataset = dataset
        self.patch_lengths = patch_lengths
        
    def calibrate_scales(self, n_samples : int = 10,
                                    max_samps : int = 5000,
                                    target_model : Optional[BaseDenoiser] = None):
        
        """ Calibrates scales to a neural network or other denoiser of class BaseDenoiser.
            
            Assumes scale should increase monotonically with the noise level.
            
         """
        
        # validation mini-batch
        batch = next(
            iter(
                DataLoader(
                    Subset(
                        self.dataset.dataset,
                        torch.arange(max_samps, max_samps + n_samples),
                    ),
                    batch_size=n_samples,
                    shuffle=False,
                )
            )
        )
        
        val_samps = batch[0] if isinstance(batch, (tuple, list)) else batch
        val_samps = val_samps.to(self.device)

        fixed_noise = torch.randn_like(val_samps, device=self.device)

        patch_length = 3
        max_patch = 2 * self.dataset.resolution - 1
        timesteps_asc = list(reversed(self.scheduler.timesteps))
        
        patch_by_t = {}
        
        for step_idx, t in enumerate(timesteps_asc):
            t_val = int(t.item())

            alpha_prod_t = self.scheduler.alphas_cumprod[t_val].to(val_samps.device)
            beta_prod_t = 1 - alpha_prod_t
            corrupted_samps = alpha_prod_t.sqrt() * val_samps + beta_prod_t.sqrt() * fixed_noise

            estimates = self._denoise_impl(
                corrupted_samps,
                t,
                patch_length=patch_length,
                max_samps=max_samps,
            )
            
            if target_model is not None:
                target_estimates = target_model.denoise(corrupted_samps, t)
            else:
                target_estimates = val_samps

            best_err = F.mse_loss(estimates, target_estimates)
            best_patch_length = patch_length
            test_patch_length = patch_length + 2

            while test_patch_length <= max_patch:
                estimates = self._denoise_impl(
                    corrupted_samps,
                    t,
                    patch_length=test_patch_length,
                    max_samps=max_samps,
                )
                err = F.mse_loss(estimates, target_estimates)
                if err <= best_err:
                    best_err = err
                    best_patch_length = test_patch_length
                    test_patch_length += 2
                else:
                    break

            patch_length = best_patch_length
            patch_by_t[t_val] = best_patch_length
        
        # set patch_lengths
        self.patch_lengths = patch_by_t
        
        return self.patch_lengths

    def calibrate_scales_generative(
        self,
        n_samples: int = 10,
        max_samps: int = 5000,
        target_model: Optional[BaseDenoiser] = None,
    ):
        """Calibrate patch scales against a target denoiser along its sampling trajectory."""

        if target_model is None:
            raise ValueError("target_model must be provided for calibrate_scales_generative")

        original_target_steps = target_model.num_steps
        target_model.set_timesteps(self.num_steps)

        with torch.no_grad():
            target_samples = target_model.sample(
                num_samples=n_samples,
                batch_size=n_samples,
                return_intermediates=True,
            )

        target_model.set_timesteps(original_target_steps)

        if target_samples.trajectory_xt is None or target_samples.timesteps is None:
            raise RuntimeError("target_model.sample(..., return_intermediates=True) must return trajectories")

        xt_by_t = {
            int(t): xt.to(self.device)
            for t, xt in zip(target_samples.timesteps, target_samples.trajectory_xt)
        }

        patch_length = 3
        max_patch = 2 * self.dataset.resolution - 1
        timesteps_asc = list(reversed(self.scheduler.timesteps))
        patch_by_t = {}

        for t in timesteps_asc:
            t_val = int(t.item())
            if t_val not in xt_by_t:
                continue

            latents_t = xt_by_t[t_val]

            with torch.no_grad():
                target_estimates = target_model.denoise(latents_t, t)

            estimates = self._denoise_impl(
                latents_t,
                t,
                patch_length=patch_length,
                max_samps=max_samps,
            )

            best_err = F.mse_loss(estimates, target_estimates)
            best_patch_length = patch_length
            test_patch_length = patch_length + 2

            while test_patch_length <= max_patch:
                estimates = self._denoise_impl(
                    latents_t,
                    t,
                    patch_length=test_patch_length,
                    max_samps=max_samps,
                )
                err = F.mse_loss(estimates, target_estimates)
                if err <= best_err:
                    best_err = err
                    best_patch_length = test_patch_length
                    test_patch_length += 2
                else:
                    break

            patch_length = best_patch_length
            patch_by_t[t_val] = best_patch_length

        self.patch_lengths = patch_by_t
        return self.patch_lengths

            
    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        patch_length : Optional[int] = None,
        label : int = None,
        **kwargs: Any,
    ) -> torch.Tensor:
    
        t_idx = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)

        if patch_length is None:
            if self.patch_lengths is None or t_idx not in self.patch_lengths:
                raise ValueError("patch_length is None and no calibrated scale was found for this timestep")
            patch_length = self.patch_lengths[t_idx]
        
        return self._denoise_impl(latents, timestep, patch_length=patch_length, label=label, **kwargs)
        
    def _denoise_impl(self, latents: torch.Tensor,
                            timestep: torch.Tensor,
                            patch_length: Optional[int] = None,
                            label: Optional[int] = None,
                            max_samps: Optional[int] = None):
        
        raise NotImplementedError

@register_model("ls_machine")
class LocalScoreMachine(LocalScoreMachineBase):
    """Local Score (LS) Machine from https://arxiv.org/pdf/2412.20292"""


    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        patch_lengths: list[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset,
            device,
            num_steps,
            params = params,
            patch_lengths = patch_lengths,
            **kwargs,
        )

    def _denoise_impl(self, latents: torch.Tensor,
                            timestep: torch.Tensor,
                            patch_length: Optional[int] = None,
                            label: Optional[int] = None,
                            max_samps : Optional[int] = None):
        
        device = latents.device
        if max_samps is None:
            max_samps = self.default_max_samps

        t_idx = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        if patch_length is None:
            if self.patch_lengths is None or t_idx not in self.patch_lengths:
                raise ValueError("patch_length is None and no calibrated scale was found for this timestep")
            patch_length = self.patch_lengths[t_idx]
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(self.device)
        beta_prod_t = 1 - alpha_prod_t
        
        b,c,h,w = latents.shape
        
        numerator = torch.zeros(latents.shape, device=device)
        denominator = torch.zeros(b,h,w, device=device)

        subtraction = None

        samples_count = 0
        max_args = None

        for x0_batch in self.dataset.dataloader:
            images = x0_batch[0] if isinstance(x0_batch, (tuple, list)) else x0_batch
            labels = x0_batch[1] if isinstance(x0_batch, (tuple, list)) else None

            if label is not None and labels is not None:
                images = images[(labels==label).squeeze(),:,:,:]

            if images.shape[0] == 0:
                continue

            images = images.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            
            if max_samps is not None:
                remaining = max_samps - samples_count
                if remaining <= 0:
                    break
                images = images[:remaining]
            
            bsize = images.shape[0]
            samples_count += bsize


            pwise_diffs = latents[:,None,:,:,:]-(alpha_prod_t**0.5)*images[None,:,:,:,:]
            pwise_normsquares = torch.sum(pwise_diffs**2, dim=2)
            patches = F.unfold(pwise_normsquares, patch_length, stride=1, padding=patch_length//2)
            patches = patches.view(b, bsize, patch_length**2, h, w)
            exp_args = -torch.sum(patches, dim=2)/(2*beta_prod_t)

            max_args = torch.maximum(max_args, torch.max(exp_args, dim=1).values) if max_args is not None else torch.max(exp_args, dim=1).values

            if subtraction is None:
                subtraction = torch.amax(exp_args, dim=1, keepdim=True)
            else:
                new_subtraction = torch.amax(exp_args, dim=1, keepdim=True)
                delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
                numerator /= torch.exp(delta_subtraction-subtraction)
                denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:]
                subtraction = delta_subtraction

            exp_vals = torch.exp(exp_args - subtraction)

            numerator = numerator + torch.sum(exp_vals[:,:,None,:,:]*images, dim=1)
            denominator = denominator + torch.sum(exp_vals, dim=1)

        return numerator / denominator[:,None]

@register_model("bbels_machine")
class BBELSMachine(LocalScoreMachineBase):
    """Boundary-Broken Equivariant Local Score (ELS) Machine from https://arxiv.org/pdf/2412.20292"""


    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        patch_lengths: list[int] = None,
        **kwargs: Any,
    ) -> None:
    
        super().__init__(
            dataset,
            device,
            num_steps,
            params = params,
            patch_lengths = patch_lengths,
            **kwargs,
        )

        self.dataset = dataset
        self.local_score_module = LocalScoreMachine(
            dataset,
            device,
            num_steps,
            params=params,
            patch_lengths=patch_lengths,
        )
    
    
    @torch.no_grad()
    def _denoise_impl(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        patch_length: Optional[int] = None,
        label: Optional[int] = None,
        max_samps: Optional[int] = None,
    ):
        device = latents.device
        if max_samps is None:
            max_samps = self.default_max_samps

        t_idx = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        if patch_length is None:
            if self.patch_lengths is None or t_idx not in self.patch_lengths:
                raise ValueError("patch_length is None and no calibrated scale was found for this timestep")
            patch_length = self.patch_lengths[t_idx]

        k = patch_length
        assert k % 2 == 1

        r = k // 2
        b, c, h, w = latents.shape

        if k >= h:
            return self.local_score_module._denoise_impl(
                latents, timestep, patch_length=patch_length, label=label
            )

        t_idx = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(device)
        beta_prod_t = 1 - alpha_prod_t
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()

        inner_y = slice(r, h - r)
        inner_x = slice(r, w - r)

        latent_norms = F.unfold(
            F.pad(latents, (r, r, r, r), value=0),
            k,
            stride=1,
        ).square().sum(dim=1).reshape(b, h, w)

        numerator = torch.zeros_like(latents)
        denominator = torch.zeros(b, h, w, device=device)
        subtraction = torch.full((b, h, w), -float("inf"), device=device)

        def add_region(ys, xs, args, vals):
            old = subtraction[:, ys, xs]
            new = torch.maximum(old, args.amax(dim=1))
            scale = torch.exp(old - new)

            numerator[:, :, ys, xs] = (
                numerator[:, :, ys, xs] * scale[:, None]
                + (torch.exp(args - new[:, None])[:, :, None] * vals).sum(dim=1)
            )
            denominator[:, ys, xs] = (
                denominator[:, ys, xs] * scale
                + torch.exp(args - new[:, None]).sum(dim=1)
            )
            subtraction[:, ys, xs] = new

        def add_center(images):
            patches = F.unfold(images, k, stride=1)
            patches = patches.permute(2, 0, 1).reshape(-1, c, k, k)

            patch_norms = patches.square().sum(dim=(1, 2, 3))
            patch_centers = patches[:, :, r, r]
            dots = F.conv2d(latents, patches, padding=0)

            args = -(
                latent_norms[:, None, inner_y, inner_x]
                - 2 * sqrt_alpha_prod_t * dots
                + alpha_prod_t * patch_norms[None, :, None, None]
            ) / (2 * beta_prod_t)

            vals = patch_centers[None, :, :, None, None].expand(
                b, -1, -1, h - 2 * r, w - 2 * r
            )

            add_region(inner_y, inner_x, args, vals)

        def add_corner(images, y0, y1, x0, x1, pad, ys, xs):
            latent_pad = F.pad(latents[:, :, y0:y1, x0:x1], pad)
            image_pad = F.pad(images[:, :, y0:y1, x0:x1], pad)

            diffs = latent_pad[:, None] - sqrt_alpha_prod_t * image_pad[None]
            args = F.unfold(diffs.square().sum(dim=2), k, stride=1)
            args = args.view(b, image_pad.shape[0], k * k, r, r)
            args = -args.sum(dim=2) / (2 * beta_prod_t)

            vals = image_pad[None, :, :, r:2*r, r:2*r].expand(b, -1, -1, -1, -1)
            add_region(ys, xs, args, vals)

        def add_edge(images, latent_edge, image_edge, edge_norms, ys, xs, transpose=False):
            L = edge_norms.shape[-1]
            image_b = image_edge.shape[0]

            args = torch.empty(b, image_b * L, r, L, device=device)
            vals = torch.empty(b, image_b * L, c, r, L, device=device)

            for j in range(r):
                latent_slice = latent_edge[:, :, j:j+k, :]
                image_slice = image_edge[:, :, j:j+k, :]

                filters = torch.cat(
                    [image_slice[:, :, :, a:a+k] for a in range(L)],
                    dim=0,
                )

                dots = F.conv2d(latent_slice, filters, padding=0)
                filter_norms = filters.square().sum(dim=(1, 2, 3))

                args[:, :, j, :] = -(
                    edge_norms[:, j, :][:, None]
                    - 2 * sqrt_alpha_prod_t * dots[:, :, 0, :]
                    + alpha_prod_t * filter_norms[None, :, None]
                ) / (2 * beta_prod_t)

                vals[:, :, :, j, :] = filters[None, :, :, r, r, None].expand(
                    b, -1, -1, L
                )

            if transpose:
                args = args.transpose(-2, -1)
                vals = vals.transpose(-2, -1)

            add_region(ys, xs, args, vals)

        dataset = self.dataset
        samples_count = 0

        for x0_batch in dataset.dataloader:
        
            if isinstance(x0_batch, (tuple, list)):
                images, labels = x0_batch[0], x0_batch[1]
            else:
                images, labels = x0_batch, None

            if label is not None and labels is not None:
                label_value = int(label.item()) if isinstance(label, torch.Tensor) else label
                images = images[labels.view(-1) == label_value]

            if images.shape[0] == 0:
                continue

            if max_samps is not None:
                remaining = max_samps - samples_count
                if remaining <= 0:
                    break
                images = images[:remaining]

            images = images.to(device)
            samples_count += images.shape[0]

            add_center(images)

            add_corner(images, 0, k - 1, 0, k - 1, (r, 0, r, 0),
                       slice(0, r), slice(0, r))
            add_corner(images, 0, k - 1, w - k + 1, w, (0, r, r, 0),
                       slice(0, r), slice(w - r, w))
            add_corner(images, h - k + 1, h, 0, k - 1, (r, 0, 0, r),
                       slice(h - r, h), slice(0, r))
            add_corner(images, h - k + 1, h, w - k + 1, w, (0, r, 0, r),
                       slice(h - r, h), slice(w - r, w))

            add_edge(
                images,
                F.pad(latents[:, :, :k-1, :], (0, 0, r, 0)),
                F.pad(images[:, :, :k-1, :], (0, 0, r, 0)),
                latent_norms[:, :r, inner_x],
                slice(0, r),
                inner_x,
            )

            add_edge(
                images,
                F.pad(latents[:, :, :, w-k+1:w], (0, r, 0, 0)).transpose(-2, -1),
                F.pad(images[:, :, :, w-k+1:w], (0, r, 0, 0)).transpose(-2, -1),
                latent_norms[:, inner_y, w-r:w].transpose(-2, -1),
                inner_y,
                slice(w - r, w),
                transpose=True,
            )

            add_edge(
                images,
                F.pad(latents[:, :, h-k+1:h, :], (0, 0, 0, r)),
                F.pad(images[:, :, h-k+1:h, :], (0, 0, 0, r)),
                latent_norms[:, h-r:h, inner_x],
                slice(h - r, h),
                inner_x,
            )

            add_edge(
                images,
                F.pad(latents[:, :, :, :k-1], (r, 0, 0, 0)).transpose(-2, -1),
                F.pad(images[:, :, :, :k-1], (r, 0, 0, 0)).transpose(-2, -1),
                latent_norms[:, inner_y, :r].transpose(-2, -1),
                inner_y,
                slice(0, r),
                transpose=True,
            )

        return numerator / denominator[:, None]

@register_model("els_machine")
class ELSMachine(LocalScoreMachineBase):
    """(Fully) Equivariant Local Score (ELS) Machine from https://arxiv.org/pdf/2412.20292"""


    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        patch_lengths: list[int] = None,
        **kwargs: Any,
    ) -> None:
    
        super().__init__(
            dataset,
            device,
            num_steps,
            params = params,
            patch_lengths = patch_lengths,
            **kwargs,
        )

        self.dataset = dataset
    
    @staticmethod
    def _circular_convolution(input_signal, kernel):
        pad_h = kernel.size(2) // 2
        pad_w = kernel.size(3) // 2
        
        input_padded = F.pad(input_signal, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        
        result = F.conv2d(input_padded, kernel, padding=0)
        
        return result

    
    @torch.no_grad()
    def _denoise_impl(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        patch_length: Optional[int] = None,
        label: Optional[int] = None,
        max_samps: Optional[int] = None,
    ):
    
        device = latents.device
        if max_samps is None:
            max_samps = self.default_max_samps

        dtype = latents.dtype

        t_idx = int(timestep.flatten()[0].item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        if patch_length is None:
            if self.patch_lengths is None or t_idx not in self.patch_lengths:
                raise ValueError("patch_length is None and no calibrated scale was found for this timestep")
            patch_length = self.patch_lengths[t_idx]

        b, c, h, w = latents.shape
        k = patch_length
        r = k // 2

        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(device=device, dtype=dtype)
        beta_prod_t = 1 - alpha_prod_t
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()

        latents_padded = F.pad(latents, (r, r, r, r), mode="circular")

        latent_norms = F.unfold(latents_padded, k, stride=1)
        latent_norms = latent_norms.square().sum(dim=1).reshape(b, h, w)

        numerator = torch.zeros_like(latents)
        denominator = torch.zeros(b, h, w, device=device, dtype=dtype)
        subtraction = None

        dataloader = self.dataset.dataloader if hasattr(self, "dataset") else self.trainloader
        samples_count = 0

        for x0_batch in dataloader:
            if isinstance(x0_batch, (tuple, list)):
                images, labels = x0_batch[0], x0_batch[1]
            else:
                images, labels = x0_batch, None

            if label is not None and labels is not None:
                label_value = int(label.item()) if isinstance(label, torch.Tensor) else label
                images = images[labels.view(-1) == label_value]

            if images.shape[0] == 0:
                continue

            if max_samps is not None:
                remaining = max_samps - samples_count
                if remaining <= 0:
                    break
                images = images[:remaining]

            images = images.to(device=device, dtype=dtype)
            samples_count += images.shape[0]

            images_padded = F.pad(images, (r, r, r, r), mode="circular")
            patches = F.unfold(images_padded, k, stride=1, padding=0)

            patches = patches.permute(2, 0, 1).reshape(-1, c, k, k)
            patch_norms = patches.square().sum(dim=(1, 2, 3))
            patch_centers = patches[:, :, r, r]

            dot_products = ELSMachine._circular_convolution(latents, patches)

            exp_args = -(
                latent_norms[:, None]
                - 2 * sqrt_alpha_prod_t * dot_products
                + alpha_prod_t * patch_norms[None, :, None, None]
            ) / (2 * beta_prod_t)

            batch_subtraction = exp_args.amax(dim=1, keepdim=True)

            if subtraction is None:
                subtraction = batch_subtraction
            else:
                new_subtraction = torch.maximum(subtraction, batch_subtraction)
                scale = torch.exp(subtraction - new_subtraction)
                numerator *= scale
                denominator *= scale[:, 0]
                subtraction = new_subtraction

            weights = torch.exp(exp_args - subtraction)

            numerator = numerator +  torch.sum(
                weights[:, :, None] * patch_centers[None, :, :, None, None],
                dim=1,
            )
            denominator = denominator + torch.sum(weights, dim=1)

        return numerator / denominator[:, None]
