# https://huggingface.co/blog/annotated-diffusion
import torch


def cosine_beta_schedule(timesteps: int, s: float=0.008) -> torch.Tensor:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def _sqrt_cumprod_alphas(timesteps: int, s: float = 1e-5) -> torch.Tensor:
    t_over_T = torch.linspace(0, 1 - s - 1e-7, timesteps)
    cumprod_one_minus_beta = 1 - torch.sqrt(t_over_T + s)
    return cumprod_one_minus_beta


def cumprod_alphas_to_betas(cumprod_one_minus_beta: torch.Tensor) -> torch.Tensor:
    betas = [1 - cumprod_one_minus_beta[0]]
    product = 1 - betas[0]
    timesteps = len(cumprod_one_minus_beta)
    for t in range(1, timesteps):
        beta = 1 - (
            cumprod_one_minus_beta[t]
            / product
        )
        betas.append(beta)
        product *= (1 - beta)
    return torch.tensor(betas)


def sqrt_beta_schedule(timesteps: int, s: float = 1e-5) -> torch.Tensor:
    """
    https://arxiv.org/pdf/2205.14217.pdf
    """
    cumprod_one_minus_beta = _sqrt_cumprod_alphas(timesteps, s)
    return cumprod_alphas_to_betas(cumprod_one_minus_beta)


BETA_SCHEDULES = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sqrt": sqrt_beta_schedule,
}
