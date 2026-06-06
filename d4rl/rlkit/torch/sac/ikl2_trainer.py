from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.optim as optim
from rlkit.torch.sac.policies import *
import torch.nn as nn
import torch
from torch.autograd import Variable, grad
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.util.io import collect_file_folder
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict
import os

EPS = np.finfo(np.float32).eps
MEAN_MIN = -9.0
MEAN_MAX = 9.0

class IKL2Trainer(TorchTrainer):
    def __init__(self,
                 env,
                 policy,
                 qf1,
                 qf2,
                 vf=None,
                 target_qf1=None,
                 target_qf2=None,

                 discount=0.99,
                 reward_scale=1.0,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_class=optim.Adam,
                 soft_target_tau=1e-2,

                 policy_update_period=1,
                 q_update_period=1,
                 target_update_period=1,
                 bc_warm_start=0,
                 total_training_steps=1E6,

                 f_reg=1.0,
                 reward_bonus=5.0,
                 alpha=0.03,
                 tau=0.9,
                 quantile=0.9,
                 l_clip=-1.0,
                 u_clip=None,

                 cosine_lr_decay=False,
                 clip_score=None,

                 bc_type='TanhGaussianPolicy',
                 bc_kwargs=dict(hidden_sizes=(256, 256)),
                 bc_norm='False',
                 use_best=False,
                 log_dir=None,
                 ):

        super().__init__()


        self.env = env
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf

        self.bc_type = bc_type
        self.bc_norm = bc_norm
        self.use_best = use_best
        self.log_dir = log_dir
        bc_checkpoint = self.bc_checkpoint()
        self.obs_dim = self.env.observation_space.low.size
        self.action_dim = self.env.action_space.low.size
        self.bc_agent = eval(self.bc_type)(obs_dim=self.obs_dim, action_dim=self.action_dim, **bc_kwargs)
        self.bc_agent.to(ptu.device)
        self.bc_agent.load_state_dict(torch.load(bc_checkpoint))

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr
        )
        self.total_training_steps = total_training_steps
        self.cosine_lr_decay = cosine_lr_decay
        if self.cosine_lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer,
                                                                           int(self.total_training_steps))

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr
        )

        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr
        )

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=qf_lr
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau

        self.policy_update_period = policy_update_period
        self.q_update_period = q_update_period
        self.target_update_period = target_update_period
        self._n_train_steps_total = 0
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics=True
        self.bc_warm_start = bc_warm_start

        self.f_reg = f_reg
        self.reward_bonus = reward_bonus
        self.alpha = alpha # additional coefficient multiplied in the log_prob ratio of current state-action
        self.tau = tau # coefficient for the KL term of next state with behavior policy
        self.quantile = quantile
        self.l_clip = float(l_clip)
        self.u_clip = u_clip
        self.clip_score = clip_score or 100.0

    def bc_checkpoint(self):
        bc_trainer_dir = os.path.join(self.log_dir, self.env.spec.id, 'BC')
        file_dir_list, num_files = collect_file_folder(bc_trainer_dir, self.bc_type+'_s_norm={}'.format(self.bc_norm))
        if num_files == 0:
            raise ValueError('No such type of BC policy')
        elif num_files > 1:
            raise ValueError('More than one BC policy, please specify the one should be chosen')
        else:
            print('will load BC agent from: \'{}\''.format(file_dir_list[0]))
            if self.use_best:
                return os.path.join(file_dir_list[0], 'best_policy.pth')
            else:
                return os.path.join(file_dir_list[0], 'final_policy.pth')

    def _get_tensor_values(self, obs, actions, network=None):
        action_size = actions.shape[0]
        obs_size= obs.shape[0]
        num_repeat = int (action_size / obs_size)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions=1, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        actions_dist = network(obs_temp)
        normal_mean = actions_dist.normal_mean # no tanh()
        normal_std = actions_dist.stddev
        new_obs_actions, log_pi = actions_dist.rsample_and_logprob() # (bs, act_dim), (bs,)
        new_obs_actions = new_obs_actions.clamp(self.min_action - 1e-6, self.max_action + 1e-6) # only useful for Gaussian dist samples

        return new_obs_actions, normal_mean, normal_std, log_pi.view(log_pi.shape[0], 1)

    def grad_penalty(self, obs, actions, network=None):
        var_actions = Variable(actions, requires_grad=True)
        var_q = network(obs, var_actions)
        ones = torch.ones(var_q.size()).to(ptu.device)

        gradient = grad(
            outputs=var_q,
            inputs=var_actions,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0] + EPS

        grad_penalty = (gradient.norm(2, dim=1)).pow(2).mean()
        return grad_penalty

    def bc_log_prob(self, obs, actions):
        actions_dist = self.bc_agent(obs)
        bc_log_actions = actions_dist.log_prob(actions)
        return bc_log_actions.unsqueeze(-1)

    def train_from_torch(self, batch):

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        QF Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob() # (bs, act_dim), (bs,)
        log_actions = dist.log_prob(actions) # (bs,)
        log_actions = log_actions.unsqueeze(-1) # (bs, 1)
        bc_log_actions = self.bc_log_prob(obs, actions)
        log_prob_ratio = (log_actions - bc_log_actions).clamp(min=self.l_clip, max=self.u_clip) # (bs, 1)

        next_target_vf = self.vf(next_obs).detach()
        q_target = self.reward_scale * rewards + self.reward_bonus + self.tau * self.alpha * log_prob_ratio \
                   + (1.0 - terminals) * self.discount * next_target_vf
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        qf1_loss = nn.MSELoss()(q1_pred, q_target)
        qf2_loss = nn.MSELoss()(q2_pred, q_target)

        # gradient regularization loss
        q1_grad_reg = self.grad_penalty(obs=obs, actions=new_obs_actions, network=self.qf1)
        q2_grad_reg = self.grad_penalty(obs=obs, actions=new_obs_actions, network=self.qf2)
        q_grad_reg = q1_grad_reg + q2_grad_reg

        total_q_loss = qf1_loss + qf2_loss + self.f_reg * q_grad_reg

        """
        VF Loss
        """
        target_q_pred = torch.min(self.target_qf1(obs, actions), self.target_qf2(obs, actions))
        neg_log_prob_ratio = (bc_log_actions - log_actions).clamp(min=self.l_clip, max=self.u_clip)
        kl_target_q_pred = target_q_pred + self.tau * neg_log_prob_ratio
        kl_target_q_pred = kl_target_q_pred.detach()

        vf_pred = self.vf(obs)
        vf_err = vf_pred - kl_target_q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        """
        Policy Loss
        """
        adv = target_q_pred - vf_pred
        exp_adv = torch.exp(adv / self.tau)
        exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        weights = exp_adv.detach()
        policy_loss = (-log_actions * weights).mean()

        """
        Update AC
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
            total_q_loss.backward()
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.cosine_lr_decay:
            self.lr_scheduler.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Q_grad_reg Loss'] = np.mean(ptu.get_numpy(q_grad_reg))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'log_prob_ratio',
                ptu.get_numpy(log_prob_ratio)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))

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
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_qf1,
            self.target_qf2,
            self.bc_agent
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            vf=self.vf,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            bc_agent=self.bc_agent
        )


