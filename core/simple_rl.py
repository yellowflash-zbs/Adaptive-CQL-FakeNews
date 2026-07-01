# coding: utf-8
"""Minimal neural-network components for this project's CQL experiments.

This replaces the large vendored rlkit dependency with the tiny subset we
actually use: MLP Q networks, a tanh-Gaussian policy, and tensor helpers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


device = torch.device("cpu")


def set_gpu_mode(mode, gpu_id=0):
    global device
    use_cuda = bool(mode) and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")


def from_numpy(array):
    return torch.from_numpy(array).float().to(device)


def get_numpy(tensor):
    return tensor.detach().cpu().numpy()


def zeros(*sizes, requires_grad=False):
    return torch.zeros(*sizes, device=device, requires_grad=requires_grad)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise ValueError("Shape must have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=lambda x: x,
    ):
        super().__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        in_size = input_size
        for idx, hidden_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, hidden_size)
            fanin_init(fc.weight)
            fc.bias.data.fill_(0.0)
            setattr(self, f"fc{idx}", fc)
            self.fcs.append(fc)
            in_size = hidden_size

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0.0)

    def forward(self, inputs):
        h = inputs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class ConcatMlp(Mlp):
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs):
        return super().forward(torch.cat(inputs, dim=self.dim))


class TanhNormal:
    def __init__(self, normal_mean, normal_std):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = torch.distributions.Independent(
            torch.distributions.Normal(normal_mean, normal_std),
            reinterpreted_batch_ndims=1,
        )

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std

    def _sample_pre_tanh(self):
        return self.normal.rsample()

    def sample(self):
        return torch.tanh(self._sample_pre_tanh()).detach()

    def rsample(self):
        return torch.tanh(self._sample_pre_tanh())

    def log_prob(self, value, pre_tanh_value=None):
        value = torch.clamp(value, -0.999999, 0.999999)
        if pre_tanh_value is None:
            pre_tanh_value = 0.5 * (torch.log1p(value) - torch.log1p(-value))
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = -2.0 * (
            torch.log(torch.tensor(2.0, device=value.device))
            - pre_tanh_value
            - F.softplus(-2.0 * pre_tanh_value)
        ).sum(dim=1)
        return log_prob + correction

    def rsample_and_logprob(self):
        pre_tanh_value = self._sample_pre_tanh()
        value = torch.tanh(pre_tanh_value)
        return value, self.log_prob(value, pre_tanh_value).view(-1, 1)


class TanhGaussianPolicy(Mlp):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -5

    def __init__(self, hidden_sizes, obs_dim, action_dim, std=None, init_w=1e-3):
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
        )
        self.std = std
        self.log_std = None
        if std is None:
            last_hidden_size = hidden_sizes[-1] if hidden_sizes else obs_dim
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)

    def forward(self, obs):
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = torch.clamp(self.last_fc_log_std(h), self.LOG_SIG_MIN, self.LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.full_like(mean, float(self.std))
        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        return TanhNormal(mean, std).log_prob(action).view(-1, 1)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0], {}

    def get_actions(self, obs_np):
        obs = from_numpy(obs_np)
        with torch.no_grad():
            dist = self(obs)
            actions = dist.sample()
        return get_numpy(actions)
