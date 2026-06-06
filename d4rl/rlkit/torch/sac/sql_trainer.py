"""
reproduce the offline rl algorithm named 'SQL'
-- 'Sparse Q-learning: in-sample offline RL via implicit value regularization'
"""

from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.core.logging import add_prefix
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict



class SQLTrainer(TorchTrainer):
    def __init__(self,
                 env,
                 policy,
                 qf1,
                 qf2,
                 vf,
                 target_qf1=None,
                 target_qf2=None,

                 discount=0.99,
                 reward_scale=1.0,

                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 optimizer_class=optim.Adam,

                 policy_update_period=1,
                 q_update_period=1,
                 target_update_period=1,

                 clip_score=100.,
                 soft_target_tau=5e-3,
                 alpha=2.0,
                 total_training_steps=1E6,
                 cosine_lr_decay=False
                 ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True


        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()


        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.total_training_steps = total_training_steps
        self.cosine_lr_decay = cosine_lr_decay
        if self.cosine_lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer,
                                                                           int(self.total_training_steps))

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=qf_lr,
        )

        self._n_train_steps_total = 0
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.alpha = alpha
        self.clip_score = clip_score


    def train_from_torch(self, batch, train=True, pretrain=False,):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        target_vf_pred = self.vf(next_obs).detach()

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_vf_pred
        q_target = q_target.detach()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss
        """
        q_pred = torch.min(
            self.target_qf1(obs, actions),
            self.target_qf2(obs, actions),
        ).detach()
        vf_pred = self.vf(obs)
        # vf_err = 1 + 0.5 * (q_pred - vf_pred) / self.alpha # original obj described in paper
        vf_err = (q_pred - vf_pred) / self.alpha + 0.5 # modified obj used in source code
        vf_sign = (vf_err > 0).float()
        vf_loss = vf_sign * (vf_err ** 2) + vf_pred / self.alpha
        vf_loss = vf_loss.mean()

        """
        Policy Loss
        """
        policy_logpp = dist.log_prob(actions)
        # weights = (vf_sign * vf_err)[:, 0].detach() # original obj described in paper
        weights = (q_pred - vf_pred).squeeze().detach() # modified obj used in source code
        if self.clip_score is not None:
            # weights = torch.clamp(weights, max=self.clip_score) # obj described in paper
            weights = torch.clamp(weights, min=0., max=self.clip_score) # obj in source code
        policy_loss = (-policy_logpp * weights).mean()

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

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
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'terminals',
                ptu.get_numpy(terminals),
            ))
            # self.eval_statistics['replay_buffer_len'] = self.replay_buffer._size
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            self.eval_statistics.update(policy_statistics)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'V1 Predictions',
                ptu.get_numpy(vf_pred),
            ))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))

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
            self.target_qf1,
            self.target_qf2,
            self.vf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            vf=self.vf,
        )
