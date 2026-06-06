import torch
from rlkit.torch.sac.policies import *
import gym, d4rl
import numpy as np
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.optim as optim
from collections import OrderedDict
from rlkit.torch.sac.policies import *

class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1=None,
            qf2=None,
            vf=None,
            target_qf1=None,
            target_qf2=None,
            policy_lr=1E-3,
            total_training_steps=1E6,
            # (mix) gaussian bc
            with_entropy_target=True,
            lr_decay=True,
            # cvae bc
            beta=0.5,
            scheduler=False,
            gamma=0.95,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.total_training_steps = total_training_steps
        self._n_train_steps_total = 0

        self.policy_lr = policy_lr
        # used for (mix)Gaussian policy
        self.with_entropy_target = with_entropy_target
        self.lr_decay = lr_decay
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])
        # used for cvae policy
        self.beta = beta
        self.scheduler = scheduler
        self.gamma = gamma

        if isinstance(self.policy, TanhGaussianPolicy):
            self.policy_type = 'single'
            self.init_gaussian_optimizer()
        elif isinstance(self.policy, VaePolicy):
            self.policy_type = 'cvae'
            self.init_cvae_optimizer()
        else:
            raise ValueError('not support such bc policy now')

        # if self.with_entropy_target:
        #     self.log_alpha = ptu.zeros(1, requires_grad=True)
        #     self.target_entropy = -np.prod(self.env.action_space.shape).item()
        #
        # if self.lr_decay:
        #     assert total_training_steps == 1E6, 'if decay policy learning rate, training step doesn\'t match'
        #     self.pi_optimizer = optim.Adam(
        #         self.policy.parameters(),
        #         lr=self.policy_lr
        #     )
        #     self.pi_scheduler = optim.lr_scheduler.MultiStepLR(
        #         self.pi_optimizer,
        #         milestones=[800000, 900000],
        #         gamma=0.1
        #     )
        #
        #     if self.with_entropy_target:
        #         self.alpha_optimizer = optim.Adam(
        #             [self.log_alpha],
        #             lr=self.policy_lr
        #         )
        #         self.alpha_scheduler = optim.lr_scheduler.MultiStepLR(
        #             self.alpha_optimizer,
        #             milestones=[800000, 900000],
        #             gamma=0.1
        #         )
        # else:
        #     self.pi_optimizer = optim.Adam(
        #         self.policy.parameters(),
        #         lr=self.policy_lr,
        #     )
        #
        #     if self.with_entropy_target:
        #         self.alpha_optimizer = optim.Adam(
        #             [self.log_alpha],
        #             lr=self.policy_lr
        #         )
    def init_gaussian_optimizer(self):
        """ optimizer for (Mix)Gaussian policy"""
        if self.with_entropy_target:
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.target_entropy = -np.prod(self.env.action_space.shape).item()
            # self.alpha = self.log_alpha.exp()

        if self.lr_decay:
            assert self.total_training_steps == 1E6, 'if decay policy learning rate, training step doesn\'t match'
            self.pi_optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.policy_lr
            )
            self.pi_scheduler = optim.lr_scheduler.MultiStepLR(
                self.pi_optimizer,
                milestones=[800000, 900000],
                gamma=0.1
            )

            if self.with_entropy_target:
                self.alpha_optimizer = optim.Adam(
                    [self.log_alpha],
                    lr=self.policy_lr
                )
                self.alpha_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.alpha_optimizer,
                    milestones=[800000, 900000],
                    gamma=0.1
                )
        else:
            self.pi_optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.policy_lr,
            )

            if self.with_entropy_target:
                self.alpha_optimizer = optim.Adam(
                    [self.log_alpha],
                    lr=self.policy_lr
                )

    def init_cvae_optimizer(self):
        self.pi_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        if self.scheduler:
            self.pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.pi_optimizer, gamma=self.gamma)

    def to(self, device):
        self.policy.to(device)

    def _get_vae_actions(self, obs, actions, network=None):
        """
        obtain reconstructed actions from VAE
        """
        mean, std = network.encode(obs, actions)
        z = mean + std * torch.randn_like(std)
        u = network.decode(obs, z)
        return u, mean, std

    def _get_policy_actions(self, obs, num_actions=1, network=None):
        """
        obtain actions from (Mix)Gaussian policy
        """
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        actions_dist = network(obs_temp)
        normal_mean = actions_dist.normal_mean # no tanh()
        normal_std = actions_dist.stddev
        new_obs_actions, log_pi = actions_dist.rsample_and_logprob() # (bs, act_dim), (bs,)
        new_obs_actions = new_obs_actions.clamp(self.min_action - 1e-6, self.max_action + 1e-6) # only useful for Gaussian dist samples

        return new_obs_actions, normal_mean, normal_std, log_pi.view(log_pi.shape[0], 1)

    def gaussian_policy_update(self, obs, actions):
        new_obs_actions, policy_mean, policy_std, log_pi = self._get_policy_actions(obs, network=self.policy)

        policy_log_prob = self.policy.logprob(actions, policy_mean, policy_std)
        policy_loss = -policy_log_prob.mean()

        # update alpha and add entropy into the BC objective
        if self.with_entropy_target:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            policy_loss += (self.log_alpha.exp().detach() * log_pi).mean()

        # update policy
        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

        if self.lr_decay:
            self.pi_scheduler.step()
            if self.with_entropy_target:
                self.alpha_scheduler.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['Policy log prob'] = np.mean(ptu.get_numpy(policy_log_prob))
            self.eval_statistics['Log pi'] = np.mean(ptu.get_numpy(log_pi))

    def cvae_policy_update(self, obs, actions):
        recon, mean, std = self._get_vae_actions(obs, actions, network=self.policy)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss

        self.pi_optimizer.zero_grad()
        vae_loss.backward()
        self.pi_optimizer.step()

        if self.scheduler and (self._n_train_steps_total + 1) % 10000 == 0:
            self.pi_scheduler.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['VAE Loss'] = np.mean(ptu.get_numpy(vae_loss))
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(KL_loss))
            self.eval_statistics['Recon Loss'] = np.mean(ptu.get_numpy(recon_loss))

    def train_from_torch(self, batch):
        obs = batch['observations']
        actions = batch['actions']

        """
        Policy update and logging
        """

        if self.policy_type == 'single':
            self.gaussian_policy_update(obs, actions)
        elif self.policy_type == 'cvae':
            self.cvae_policy_update(obs, actions)
        else:
            raise ValueError('not support such bc policy now')

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy
        )

