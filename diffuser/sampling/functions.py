import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, scale=0.001, t_stopgrad=0, 
    n_guide_steps=1, scale_grad_by_std=True,
):  
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    # mdoel.posterior_log_variance_clipped: (20)
    # 提取 posterior_log_variance_clipped 中 t 索引的值，并重复 t.shape 次
    y = None

    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y
