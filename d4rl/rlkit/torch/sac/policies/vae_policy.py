import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import rlkit.torch.pytorch_util as ptu
import math
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.distributions import Delta



class VaePolicy(TorchStochasticPolicy):
    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim=None,
            max_action=1.0,
            e_hidden_sizes=(750, 750),
            d_hidden_sizes=(750, 750),
            iwae=True,
    ):
        super().__init__()
        self.encoders = []
        self.decoders = []
        self.latent_dim = latent_dim or action_dim * 2

        e_ipt_size = obs_dim + action_dim

        for i, e_next_size in enumerate(e_hidden_sizes):
            e = nn.Linear(e_ipt_size, e_next_size)
            e_ipt_size = e_next_size
            self.__setattr__("e{}".format(i), e)
            self.encoders.append(e)

        self.mean = nn.Linear(e_hidden_sizes[-1], self.latent_dim)
        self.log_std = nn.Linear(e_hidden_sizes[-1], self.latent_dim)

        d_ipt_size = obs_dim + self.latent_dim
        for i, d_next_size in enumerate(d_hidden_sizes):
            d = nn.Linear(d_ipt_size, d_next_size)
            d_ipt_size = d_next_size
            self.__setattr__("d{}".format(i), d)
            self.decoders.append(d)

        self.last_fc = nn.Linear(d_hidden_sizes[-1], action_dim)

        self.max_action = max_action
        self.iwae = iwae

    def get_stochastic_action(self, obs_np): # sample actions with certain probability
        info = None
        obs_np = obs_np.reshape(1, -1) # np: (1, obs_dim)
        action = self.decode(ptu.from_numpy(obs_np))

        action = ptu.get_numpy(action)
        return action[0, :], info

    def forward(self, obs): # return normal distribution whose mean equals to the deterministic actions from decoder
        z = ptu.zeros((obs.shape[0], self.latent_dim)) # with highest probability
        actions = self.decode(obs, z)
        return Delta(actions)

    def encode(self, state, action):
        h = torch.cat([state, action], -1)
        for i, e in enumerate(self.encoders):
            h = e(h)
            h = F.relu(h)

        mean = self.mean(h)
        # Clamped for numerical stability
        log_std = self.log_std(h).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        a = torch.cat([state, z], -1)
        for i, d in enumerate(self.decoders):
            a = d(a)
            a = F.relu(a)
        if self.max_action is not None:
            return self.max_action * torch.tanh(self.last_fc(a))
        else:
            return self.last_fc(a)

    def log_prob(self, obs, action):
        if not self.iwae:
            log_pi = -1. * self.elbo_loss(obs, action, beta=0.5, num_samples=1)
        else:
            log_pi = -1. * self.iwae_loss(obs, action, beta=0.5, num_samples=1)

        return log_pi # (bs,)

    def elbo_loss(self, obs, action, beta, num_samples=1):
        """
        Note: elbo_loss one is proportional to elbo_estimator
        i.e. there exist a>0 and b, elbo_loss = a * (-elbo_estimator) + b
        """
        mean, std = self.encode(obs, action) # (bs, z_dim)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s) # [B x S x D]

        obs = obs.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        u = self.decode(obs, z) # [B x S x C]
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss # (bs,)

    def iwae_loss(self, obs, action, beta, num_samples=10):
        ll = self.importance_sampling_estimator(obs, action, beta, num_samples)
        return -ll

    def elbo_estimator(self, obs, action, beta, num_samples=1):
        mean, std = self.encode(obs, action)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)

        obs= obs.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(obs, z)
        std_dec = math.sqrt(beta / 4)

        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(ptu.device) * std_dec
        log_pxz = Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).sum(-1)
        elbo = log_pxz.sum(-1).mean(-1) - KL_loss
        return elbo

    def importance_sampling_estimator(self, obs, action, beta, num_samples=500):
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper, we also need **an expection over L samples**
        mean, std = self.encode(obs, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        obs = obs.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(obs, z)
        std_dec = math.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(ptu.device)
        std_prior = torch.ones_like(z).to(ptu.device)
        log_pz = Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(ptu.device) * std_dec
        log_pxz = Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - math.log(num_samples)
        return ll # (bs,)
