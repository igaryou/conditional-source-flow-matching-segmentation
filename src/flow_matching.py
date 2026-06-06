from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from scipy.linalg import helmert


class ConditionalSourceFlowMatching:
    def __init__(
        self,
        K: int,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 2.0,
        pow: float = 2.0,
        lam: float = 0.5,
        alpha_1: float = 100.0,
        alpha_0: float = 10.0,
        device: torch.device | str = "cuda",
        z0_mode: str = "generator",
        sampler: str = "euler",
        use_ema_weight: bool = False,
        ema_beta: float = 0.99,
        scale_eps: float = 1.0e-8,
        weight_clip_min: float = 1.0e-3,
        weight_clip_max: float = 100.0,
        aux_warmup_steps: int = 1000,
    ) -> None:
        if abs((a + b) - 1.0) > 1.0e-6:
            raise ValueError(f"a + b must equal 1, got {a + b}")

        self.K = K
        self.a = a
        self.b = b
        self.c = c
        self.pow = pow
        self.lam = lam
        self.alpha_1 = alpha_1
        self.alpha_0 = alpha_0
        self.device = torch.device(device)
        self.z0_mode = z0_mode
        self.sampler = sampler

        self.use_ema_weight = use_ema_weight
        self.ema_beta = ema_beta
        self.scale_eps = scale_eps
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max
        self.aux_warmup_steps = aux_warmup_steps
        self.ema_mse = None
        self.ema_var = None
        self.ema_align = None
        self.ema_step = 0

        matrix = helmert(K, full=False)
        self.H = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)

    def to_open_simplex(self, categorical_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = categorical_map.shape
        concentration = torch.full(
            (channels,),
            self.alpha_1,
            device=categorical_map.device,
            dtype=torch.float32,
        )
        noise = torch.distributions.Dirichlet(concentration).sample(
            (batch, height, width)
        )
        noise = noise.permute(0, 3, 1, 2)
        simplex = self.lam * categorical_map + (1.0 - self.lam) * noise
        simplex = simplex.clamp_min(1.0e-6)
        return simplex / simplex.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def ilr(self, simplex: torch.Tensor) -> torch.Tensor:
        matrix = self.H.to(device=simplex.device, dtype=simplex.dtype)
        return torch.einsum("dk,bkhw->bdhw", matrix, torch.log(simplex))

    def decode_ilr(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix = self.H.to(device=z.device, dtype=z.dtype)
        logits = torch.einsum("kd,bdhw->bkhw", matrix.T, z)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities, probabilities.argmax(dim=1)

    def masks_to_z1(self, masks: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(masks.long(), num_classes=self.K)
        onehot = onehot.permute(0, 3, 1, 2).float()
        return self.ilr(self.to_open_simplex(onehot))

    def tau(self, t: torch.Tensor) -> torch.Tensor:
        return self.a * t + self.b * t**self.pow

    def dot_tau(self, t: torch.Tensor) -> torch.Tensor:
        return self.a + self.pow * self.b * t ** (self.pow - 1.0)

    def sample_gaussian(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            (batch, self.K - 1, height, width),
            device=device,
            dtype=dtype,
        )

    def sample_dirichlet(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        concentration = torch.full(
            (self.K,), self.alpha_0, device=device, dtype=torch.float32
        )
        simplex = torch.distributions.Dirichlet(concentration).sample(
            (batch, height, width)
        )
        simplex = simplex.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        simplex = simplex.clamp_min(1.0e-6)
        simplex = simplex / simplex.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return self.ilr(simplex)

    def sample_uniform(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        simplex = torch.full(
            (batch, self.K, height, width),
            1.0 / self.K,
            device=device,
            dtype=dtype,
        )
        return self.ilr(simplex)

    def sample_initial_z0(
        self,
        batch: int,
        height: int,
        width: int,
        source_generator: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.z0_mode in {"generator", "csfm"}:
            z0, _, _ = source_generator(image)
            return z0
        if self.z0_mode in {"gauss", "gaussian", "euclidean"}:
            return self.sample_gaussian(batch, height, width, device, dtype)
        if self.z0_mode in {"dirichlet", "random"}:
            return self.sample_dirichlet(batch, height, width, device, dtype)
        if self.z0_mode in {"uniform", "uni"}:
            return self.sample_uniform(batch, height, width, device, dtype)
        raise ValueError(
            f"Unknown z0_mode '{self.z0_mode}'. "
            "Choose generator, gaussian, dirichlet, or uniform."
        )

    def flow(
        self,
        z1: torch.Tensor,
        image: torch.Tensor,
        source_generator: torch.nn.Module,
    ) -> Tuple[torch.Tensor, ...]:
        batch = z1.shape[0]
        t = torch.rand(batch, device=z1.device, dtype=z1.dtype)
        t_view = t.view(batch, 1, 1, 1)
        z0, mu, logvar = source_generator(image)
        zt = self.tau(t_view) * z1 + (1.0 - self.tau(t_view)) * z0
        return zt, z0, t, mu, logvar

    def get_aux_warmup_factor(self) -> float:
        if self.aux_warmup_steps is None or self.aux_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.ema_step) / float(self.aux_warmup_steps))

    @torch.no_grad()
    def update_loss_ema(
        self,
        loss_mse: torch.Tensor,
        loss_var: torch.Tensor,
        loss_align: torch.Tensor,
    ) -> None:
        mse = float(loss_mse.detach().cpu())
        variance = float(loss_var.detach().cpu())
        alignment = float(loss_align.detach().cpu())
        if self.ema_mse is None:
            self.ema_mse = mse
            self.ema_var = variance
            self.ema_align = alignment
        else:
            beta = self.ema_beta
            self.ema_mse = beta * self.ema_mse + (1.0 - beta) * mse
            self.ema_var = beta * self.ema_var + (1.0 - beta) * variance
            self.ema_align = beta * self.ema_align + (1.0 - beta) * alignment
        self.ema_step += 1

    def get_dynamic_weights(
        self, rho_var: float, rho_align: float
    ) -> Tuple[float, float, float]:
        if self.ema_mse is None:
            return 0.0, 0.0, 0.0
        var_weight = rho_var * self.ema_mse / (self.ema_var + self.scale_eps)
        align_weight = (
            rho_align * self.ema_mse / (self.ema_align + self.scale_eps)
        )
        var_weight = max(
            self.weight_clip_min, min(self.weight_clip_max, var_weight)
        )
        align_weight = max(
            self.weight_clip_min, min(self.weight_clip_max, align_weight)
        )
        warmup = self.get_aux_warmup_factor()
        return warmup * var_weight, warmup * align_weight, warmup

    def loss(
        self,
        onehot: torch.Tensor,
        image: torch.Tensor,
        model: torch.nn.Module,
        source_generator: torch.nn.Module,
        var_weight: float = 0.25,
        align_weight: float = 0.25,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        onehot = onehot.to(device=image.device, dtype=image.dtype)
        z1 = self.ilr(self.to_open_simplex(onehot))
        batch = z1.shape[0]
        zt, z0, t, mu, logvar = self.flow(z1, image, source_generator)

        target_velocity = self.dot_tau(t.view(batch, 1, 1, 1)) * (z1 - z0)
        predicted_velocity = model(zt, image, t)
        loss_mse = F.mse_loss(
            target_velocity, predicted_velocity, reduction="mean"
        )

        if getattr(source_generator, "fixed_std", None) is not None:
            loss_var = torch.zeros(
                (), device=loss_mse.device, dtype=loss_mse.dtype
            )
        else:
            loss_var = 0.5 * torch.mean(torch.exp(logvar) - logvar - 1.0)

        mu_normalized = F.normalize(mu, p=2, dim=1)
        z1_normalized = F.normalize(z1, p=2, dim=1)
        loss_align = F.mse_loss(mu_normalized, z1_normalized)

        if self.use_ema_weight:
            self.update_loss_ema(loss_mse, loss_var, loss_align)
            var_weight_dynamic, align_weight_dynamic, warmup = (
                self.get_dynamic_weights(var_weight, align_weight)
            )
        else:
            var_weight_dynamic = var_weight
            align_weight_dynamic = align_weight
            warmup = 1.0

        weighted_var = var_weight_dynamic * loss_var
        weighted_align = align_weight_dynamic * loss_align
        loss = loss_mse + weighted_var + weighted_align

        with torch.no_grad():
            sigma = torch.exp(0.5 * logvar)
            logs = {
                "loss": float(loss.detach().cpu()),
                "loss_mse": float(loss_mse.detach().cpu()),
                "loss_var": float(loss_var.detach().cpu()),
                "loss_align": float(loss_align.detach().cpu()),
                "weighted_var": float(weighted_var.detach().cpu()),
                "weighted_align": float(weighted_align.detach().cpu()),
                "var_weight_dyn": float(var_weight_dynamic),
                "align_weight_dyn": float(align_weight_dynamic),
                "aux_warmup": float(warmup),
                "ema_mse": float(self.ema_mse or 0.0),
                "ema_var": float(self.ema_var or 0.0),
                "ema_align": float(self.ema_align or 0.0),
                "pred_velocity": float(
                    predicted_velocity.abs().mean().detach().cpu()
                ),
                "target_velocity": float(
                    target_velocity.abs().mean().detach().cpu()
                ),
                "zt_abs": float(zt.abs().mean().detach().cpu()),
                "z1_abs": float(z1.abs().mean().detach().cpu()),
                "mu_abs": float(mu.abs().mean().detach().cpu()),
                "mu_min": float(mu.min().detach().cpu()),
                "mu_max": float(mu.max().detach().cpu()),
                "logvar_mean": float(logvar.mean().detach().cpu()),
                "logvar_min": float(logvar.min().detach().cpu()),
                "logvar_max": float(logvar.max().detach().cpu()),
                "sigma_mean": float(sigma.mean().detach().cpu()),
                "sigma_min": float(sigma.min().detach().cpu()),
                "sigma_max": float(sigma.max().detach().cpu()),
            }
        return loss, logs

    @torch.no_grad()
    def _sample(
        self,
        method: str,
        model: torch.nn.Module,
        source_generator: torch.nn.Module,
        image: torch.Tensor,
        num_steps: int,
        return_traj: bool,
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        model.eval()
        source_generator.eval()
        batch, _, height, width = image.shape
        z = self.sample_initial_z0(
            batch=batch,
            height=height,
            width=width,
            source_generator=source_generator,
            image=image,
            device=image.device,
            dtype=image.dtype,
        )
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=image.device)
        trajectory = []

        def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            time_batch = torch.full(
                (batch,), time, device=image.device, dtype=state.dtype
            )
            return model(state, image, time_batch)

        for index in range(num_steps):
            t0 = times[index]
            t1 = times[index + 1]
            dt = t1 - t0
            half_dt = dt / 2.0
            if method == "euler":
                z = z + dt * velocity(z, t0)
            elif method == "midpoint":
                k1 = velocity(z, t0)
                k2 = velocity(z + half_dt * k1, t0 + half_dt)
                z = z + dt * k2
            elif method == "RK4":
                k1 = velocity(z, t0)
                k2 = velocity(z + half_dt * k1, t0 + half_dt)
                k3 = velocity(z + half_dt * k2, t0 + half_dt)
                k4 = velocity(z + dt * k3, t1)
                z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            if return_traj:
                _, prediction = self.decode_ilr(z)
                trajectory.append(prediction.cpu())

        probabilities, prediction = self.decode_ilr(z)
        if return_traj:
            return torch.stack(trajectory, dim=0)
        return probabilities, prediction

    @torch.no_grad()
    def sample_euler(self, **kwargs):
        return self._sample(method="euler", **kwargs)

    @torch.no_grad()
    def sample_midpoint(self, **kwargs):
        return self._sample(method="midpoint", **kwargs)

    @torch.no_grad()
    def sample_RK4(self, **kwargs):
        return self._sample(method="RK4", **kwargs)

    def sampling_method(
        self,
        model: torch.nn.Module,
        source_generator: torch.nn.Module,
        image: torch.Tensor | None = None,
        num_steps: int = 100,
        return_traj: bool = False,
        **legacy_kwargs,
    ):
        image = image if image is not None else legacy_kwargs.pop("img", None)
        if image is None:
            raise ValueError("image is required")
        kwargs = {
            "model": model,
            "source_generator": source_generator,
            "image": image,
            "num_steps": num_steps,
            "return_traj": return_traj,
        }
        if self.sampler == "euler":
            return self.sample_euler(**kwargs)
        if self.sampler == "midpoint":
            return self.sample_midpoint(**kwargs)
        if self.sampler in {"RK4", "rk4"}:
            return self.sample_RK4(**kwargs)
        raise ValueError(
            f"Unknown sampler '{self.sampler}'. Choose euler, midpoint, or RK4."
        )


sim_euFlow = ConditionalSourceFlowMatching
